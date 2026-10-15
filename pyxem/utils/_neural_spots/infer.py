# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

# Vendored from SpyDE (https://github.com/directelectron/spyde) at 68dfa52038,
# spyde/models/infer.py; see vendor note in __init__.py.

"""Parameter-free inference: scale-norm -> local norm -> model -> decode -> map back.

``load_model`` + ``detect`` are vendored from the ``yoloDiffraction`` research
project (``yolodiffraction/model/infer.py``), rewired to import from the sibling
modules in this package. ``detect_batch`` is new to SpyDE: it runs a whole stack
of frames through one forward pass (the batched path the 4D-STEM chunk compute
needs), decoding GPU-resident (``center.decode_batch_centers``).

The full single-frame pipeline a user runs needs no parameters: disk size is
estimated automatically and the frame rescaled to the canonical size before the
model; predicted positions are mapped back to the original frame coordinates.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

from .center import decode_batch_centers
from .preprocess import (
    CANONICAL_DIAMETER,
    estimate_disk_diameter,
    normalize_input,
    scale_to_canonical,
)
from .unet import SpotUNet

log = logging.getLogger(__name__)


# ── Apple-MPS robustness gates (Mac-only) ─────────────────────────────────────
# SpotUNet uses nn.ConvTranspose2d + BatchNorm; ConvTranspose on MPS has a long
# crash/correctness bug history, and the multi-worker batch path runs several
# MPS forwards concurrently across worker processes. On Mac, some MPS op failures
# raise a catchable RuntimeError, but the ones that take the *process* down are
# uncatchable native Metal aborts (SIGABRT/segfault) — no try/except catches
# those. The gates below make the Mac path non-crashing by default while keeping
# MPS for the safe single-thread preview.
#
# ── THE ESCAPE-HATCH FLAG (PYXEM_NEURAL_MPS_BATCH) ───────────────────────────
# On Mac the neural BATCH runs on MPS **by default** (validated on real Metal:
# SpotUNet's forward is correct and ~6x faster than CPU on Apple silicon, and the
# batch is now pinned to a SINGLE worker process — see orchestrate._mps_neural_lane
# — so the concurrent-Metal-context abort that motivated the old CPU default can no
# longer happen). ``PYXEM_NEURAL_MPS_BATCH`` only exists now as an OFF switch for a
# user whose specific hardware/torch build still misbehaves:
#     unset / "1" / "on"   → batch uses MPS (the default on Mac).
#     "0" / "off"          → batch forced onto CPU (the old safe default; the
#                            batched detect_batch path, just on the CPU device).
# The layered safety nets (MPS fallback env, catchable-error CPU retry, device
# serialization, worker-death → CPU demotion) all still apply. On non-Mac (CUDA)
# this gate is IRRELEVANT — ``mps_batch_allowed`` only reads the env on darwin;
# CUDA behavior is completely unchanged.
def _is_mac() -> bool:
    return sys.platform == "darwin"


def mps_batch_allowed() -> bool:
    """True iff the multi-worker neural BATCH may use MPS. Off-Mac this is always
    True (the gate is a no-op — CUDA/CPU decide via the device chain). On Mac it is
    True by default (MPS is validated + single-worker-pinned) and only False when
    ``PYXEM_NEURAL_MPS_BATCH`` is explicitly set to 0/off (the CPU escape hatch)."""
    if not _is_mac():
        return True
    return os.environ.get("PYXEM_NEURAL_MPS_BATCH", "").lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def enable_mps_cpu_fallback() -> None:
    """Set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` on Mac so an unsupported MPS op (e.g.
    a ConvTranspose gap in a given torch build) transparently runs on CPU per-op
    instead of raising or aborting. MUST be called BEFORE torch initializes MPS
    (i.e. before the first MPS tensor/op) to take effect — so this is invoked at
    process startup (main process and every worker). No-op off Mac; idempotent."""
    if not _is_mac():
        return
    # Only set if unset so a user override (e.g. explicitly "0") is respected.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# On import (main process + every worker that imports this module) make the
# unsupported-op → CPU fallback active before any MPS work. Cheap + idempotent.
enable_mps_cpu_fallback()


def _default_device():
    """Best device for inference: CUDA → Apple-MPS → CPU (mirrors
    ``find_vectors_torch.torch_gpu_device``; previously Macs silently ran the
    model on CPU while the batch path believed it had a GPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def load_model(ckpt_path, device=None, arch: dict | None = None):
    """Load a SpotUNet checkpoint. ``arch`` (from the model registry) overrides the
    architecture hyperparams when present; otherwise they're read from the
    checkpoint (which stores ``base``/``in_ch``/``levels`` and, for newer models,
    ``mask_head``).

    The operating point a model was trained for travels with it, read from the
    checkpoint onto the model: ``mad_floor`` (floor on the normalization scale, None
    for the older models) and ``default_threshold`` (the heatmap threshold
    ``calibrate`` starts from, 0.3 unless the checkpoint says otherwise; set so that
    models run at a comparable false-positive rate)."""
    device = device or _default_device()
    # weights_only=True: checkpoints are plain state dicts + scalar hyperparams;
    # never unpickle arbitrary objects from a (possibly downloaded) file.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arch = arch or {}
    in_ch = arch.get("in_ch", ck.get("in_ch", 1))
    model = SpotUNet(
        in_ch=in_ch,
        base=arch.get("base", ck["base"]),
        levels=arch.get("levels", ck.get("levels", 2)),
        mask_head=bool(arch.get("mask_head", ck.get("mask_head", False))),
    )
    model.load_state_dict(ck["state_dict"])
    mad_floor = float(ck.get("mad_floor", 0.0) or 0.0)
    model.mad_floor = mad_floor if mad_floor > 0 else None
    model.default_threshold = float(ck.get("threshold", 0.3) or 0.3)
    # Moving the weights to MPS and the smoke-test forward are both real device
    # submissions, and ``registry.get_model`` deliberately calls this OUTSIDE its
    # cache lock (so a slow download doesn't block the cache) — so several
    # find-vectors chunk threads can land here at once on a cold cache. Concurrent
    # Metal submission is an uncatchable SIGSEGV, so take THE process-wide device
    # lock for the device-touching part. Null context off MPS. See pyxem.utils._neural_spots.device_lock.
    from .device_lock import accelerator_lock

    with accelerator_lock(device):
        model = model.to(device)
        model.eval()
        if device.type == "mps":
            # Smoke-test the forward ONCE at load: an MPS build missing an op this
            # net uses should degrade to CPU here, not fail on every chunk.
            try:
                with torch.no_grad():
                    model(torch.zeros(1, in_ch, 32, 32, device=device))
            except Exception:
                device = torch.device("cpu")
                model = model.to(device)
    return model, device


def _mask_radius(work_diam: float) -> float:
    """Expected disk radius at working resolution, for the mask-centroid decode."""
    if np.isfinite(work_diam) and work_diam > 0:
        return 0.5 * float(work_diam)
    return 0.5 * CANONICAL_DIAMETER


def _pad_to_multiple(nrm: np.ndarray, levels: int):
    """Reflect-pad H,W up to a multiple of 2**levels so the U-Net pool/upsample
    line up. Returns the padded array (unchanged if already aligned)."""
    H0, W0 = nrm.shape
    mult = 2**levels
    ph = (mult - H0 % mult) % mult
    pw = (mult - W0 % mult) % mult
    if ph or pw:
        nrm = np.pad(nrm, ((0, ph), (0, pw)), mode="reflect")
    return nrm


# Local-norm high-pass scale (px). The default suits normal data; VERY diffuse
# backgrounds (beam stop / amorphous halo) need a SMALLER value — set automatically
# by ``calibrate`` / ``auto_bg_sigma`` rather than by hand.
DEFAULT_BG_SIGMA = 12.0


def _estimate_work_diam(
    frame: np.ndarray, factor: float, spot_diameter: float | None
) -> float:
    """Disk diameter at WORKING (post-scale) resolution — drives BOTH the
    background-subtraction scale and the NMS window for big disks run near-native
    (the SCALE_CLIP floor keeps very large disks well above canonical). Ported from
    yoloDiffraction detect(): the autocorrelation estimator high-passes at
    sigma=20 (canonical-tuned) and SATURATES on big disks, so iterate with a
    size-adaptive high-pass, then correct its measured ~0.83 undershoot."""
    if spot_diameter:
        return float(spot_diameter) * factor
    wd = estimate_disk_diameter(frame)
    if np.isfinite(wd) and wd > 20:
        for _ in range(3):
            wd2 = estimate_disk_diameter(frame, hp_sigma=max(20.0, wd))
            if abs(wd2 - wd) < 1:
                wd = wd2
                break
            wd = max(wd, wd2)
        wd = wd / 0.83
    return wd * factor


def _big_disk_params(md: int, bg_sigma: float | None, work_diam: float):
    """Resolve the NMS window + background sigma for the working disk size.

    Canonical-size data (work_diam <= 20) is untouched, so existing (production-
    model) behavior is unchanged; this only engages on big disks where the fixed
    values were wrong two ways (measured in yoloDiffraction, 24-96px disks):
      - a fixed ~4px NMS window fires a CLOUD of detections across one broad disk
        -> md ~ 0.55*diameter collapses it to ONE (center err <1.3px);
      - a fixed 12px background sigma high-passes INSIDE the disk (carves the flat
        interior into a ring the model fires on) -> bg ~ 0.6*diameter keeps the
        interior intact. An explicitly-passed bg_sigma (e.g. from calibrate) is
        respected; only the None default is size-scaled."""
    big = np.isfinite(work_diam) and work_diam > 20
    if big:
        md = max(md, int(round(work_diam * 0.55)))
    if bg_sigma is None:
        bg_sigma = max(DEFAULT_BG_SIGMA, 0.6 * work_diam) if big else DEFAULT_BG_SIGMA
    return md, float(bg_sigma)


@torch.no_grad()
def auto_bg_sigma(
    model, work: np.ndarray, device, candidates=(4, 5, 6, 8, 10, 12, 16, 20)
) -> tuple[float, float]:
    """Pick the local-norm ``bg_sigma`` that MAXIMIZES the model's heatmap confidence.

    The model's own peak confidence is a reliable, hands-off signal for how well the
    diffuse background was removed (confidence has a clean maximum vs bg_sigma). Used
    by ``calibrate`` on a few representative frames; reused across the scan. Returns
    (best_bg_sigma, best_confidence)."""
    levels = int(getattr(model, "levels", 2))
    best_conf, best_sigma = -1.0, DEFAULT_BG_SIGMA
    mad_floor = getattr(model, "mad_floor", None)
    for s in candidates:
        nrm = _pad_to_multiple(
            normalize_input(work, local=True, bg_sigma=float(s), mad_floor=mad_floor),
            levels,
        )
        hm = model(torch.from_numpy(nrm[None, None].astype(np.float32)).to(device))[0]
        c = float(torch.sigmoid(hm).max())
        if c > best_conf:
            best_conf, best_sigma = c, float(s)
    return best_sigma, best_conf


def bg_sigma_from_peak_size(peak_diameter_px: float) -> float:
    """Map a known/estimated peak diameter (px) -> bg_sigma (high-pass scale): a few
    x the peak so the high-pass removes the diffuse background but not the peaks."""
    return float(np.clip(1.2 * peak_diameter_px, 4.0, 24.0))


def _to_frame_coordinates(peaks: np.ndarray, frame_shape, factor: float) -> np.ndarray:
    """Map peak ``(y, x)`` from the rescaled working frame back to the frame.

    The rescale (``scipy.ndimage.zoom(order=1)`` and its torch twin) aligns the
    first and last pixel centers, so working pixel ``i`` sits at frame position
    ``i * (n - 1) / (out - 1)`` with ``out = round(n * factor)`` -- not at
    ``i / factor``. The two differ by the rounding of the output size: a uniform
    scale error of up to ~0.6 %, invisible in relative strain but a bias of that
    size in absolute strain and up to ~0.7 px at the far edge of a 128 px frame.
    Only this inverse changes; the forward rescale stays as trained."""
    if factor == 1.0 or len(peaks) == 0:
        return peaks
    peaks = peaks.copy()
    for axis, n in enumerate(frame_shape[:2]):
        out = int(round(n * factor))
        peaks[:, axis] = peaks[:, axis] * ((n - 1) / (out - 1))
    return peaks


def _width_radius(work_diam: float, md: int) -> float:
    """Window the response width is taken over (working px): the disk's radius,
    but never past the NMS distance, so a neighboring disk's response is not
    averaged in."""
    r = (
        0.5 * float(work_diam)
        if np.isfinite(work_diam) and work_diam > 0
        else float(md)
    )
    return float(max(1.5, min(r, md)))


def _frame_rows(res, frame_shape, factor: float, with_width: bool) -> np.ndarray:
    """``decode_batch_centers`` rows of one frame -> ``[y, x, score(, width)]``
    in original-frame pixels."""
    res = res.detach().cpu().numpy() if hasattr(res, "detach") else np.asarray(res)
    cols = [1, 2, 3, 4] if with_width else [1, 2, 3]
    out = _to_frame_coordinates(
        res[:, cols].astype(np.float32, copy=True), frame_shape, factor
    )
    if with_width and len(out) and factor != 1.0:
        # A width, not a position: the end-pixel rounding the inverse corrects
        # is a fraction of a percent of it.
        out[:, 3] = out[:, 3] / factor
    return out


@torch.no_grad()
def detect(
    model,
    frame: np.ndarray,
    device,
    thresh: float = 0.3,
    min_distance: int = 4,
    auto_scale: bool = True,
    bg_sigma: float | None = None,
    spot_diameter: float | None = None,
    with_width: bool = False,
):
    """Detect spots in a single frame. Returns (N,3) [y,x,score] in ORIGINAL coords
    (``with_width``: (N,4) [y,x,score,width], width in original pixels).

    Estimates disk size, rescales to canonical, runs the model, maps positions back.
    ``bg_sigma`` is the local-norm high-pass scale (set by ``calibrate`` for diffuse
    data; ``None`` = automatic: the default 12 for normal data, scaled up with the
    disk for big disks run near-native). ``thresh`` is the heatmap confidence.
    ``spot_diameter`` (px) overrides the autocorrelation disk-size estimate — the
    user-facing "Spot size" knob for when the estimate gets it wrong; the canonical
    rescale then derives from it (both directions, bounded by SCALE_CLIP).
    """
    factor = 1.0
    work = frame
    if auto_scale:
        work, factor = scale_to_canonical(frame, diameter=spot_diameter)
    md = max(2, int(round(min_distance * factor)))
    work_diam = _estimate_work_diam(frame, factor, spot_diameter)
    md, bg_sigma = _big_disk_params(md, bg_sigma, work_diam)
    nrm = normalize_input(
        work, local=True, bg_sigma=bg_sigma, mad_floor=getattr(model, "mad_floor", None)
    )
    levels = int(getattr(model, "levels", 2))
    nrm = _pad_to_multiple(nrm, levels)
    x = torch.from_numpy(nrm[None, None]).to(device)
    outputs = model(x)
    res = decode_batch_centers(
        outputs[0],
        outputs[1],
        thresh=thresh,
        min_distance=md,
        radius=_width_radius(work_diam, md),
        mask_logits=outputs[2] if len(outputs) > 2 else None,
        mask_radius=_mask_radius(work_diam),
    )
    return _frame_rows(res, frame.shape, factor, with_width)


def _neural_sub_batch_size() -> int:
    """Sub-batch size (frames per forward pass) for ``detect_batch``. A single
    forward pass over a whole nav chunk (1000+ frames) allocates activations for
    every frame at once — tens of GB for a big U-Net. Looping in fixed sub-batches
    bounds peak activation memory to O(K) instead of O(N) while keeping identical
    results (the loop is purely a memory/throughput knob; the per-frame math is
    unchanged). ``PYXEM_NEURAL_BATCH`` overrides the default; parsed once per call,
    clamped >= 1."""
    import os

    try:
        k = int(os.environ.get("PYXEM_NEURAL_BATCH", "64"))
    except ValueError:
        k = 64
    return max(1, k)


def _forward_with_cpu_retry(model, device, x):
    """Run ``model`` on the ``(K,1,H,W)`` tensor ``x``; on a torch/MPS
    ``RuntimeError`` move the model AND tensor to CPU, retry on CPU, and return the
    CPU model+device so the caller stays on CPU (fix 2 — catchable MPS op failure →
    per-op CPU retry). ``x`` may already sit on ``device`` (on-device preprocessing)
    or on CPU (host numpy prep); it is moved to the active device either way.

    Returns ``(outputs, model, device)`` — the model's output tuple (heatmap,
    offsets, and the mask for a mask-head model) and the (possibly CPU-moved)
    model+device the caller must use for the REST of this call. On CUDA/CPU the
    retry path never triggers, so behavior there is unchanged."""
    xd = x.to(device)
    try:
        return model(xd), model, device
    except RuntimeError as e:
        # A catchable MPS (or other device) op failure. Re-move the model to CPU
        # ONCE and retry this sub-batch there; subsequent sub-batches reuse the
        # CPU model. (PYTORCH_ENABLE_MPS_FALLBACK should already route unsupported
        # ops to CPU per-op; this catches the ones that still raise.)
        if device.type == "cpu":
            raise
        log.warning(
            "[models] %s forward raised (%s); retrying this sub-batch on "
            "CPU and staying on CPU for the rest of this call",
            device.type,
            e,
        )
        del xd
        cpu = torch.device("cpu")
        model = model.to(cpu)
        model.eval()
        # Also demote the SHARED registry cache so the next get_model() caller in
        # this process doesn't re-select the just-failed device and crash again.
        try:
            from .weights import demote_cached_models_to_cpu

            demote_cached_models_to_cpu()
        except Exception:
            pass
        return model(x.to(cpu)), model, cpu


def _gpu_prep_enabled() -> bool:
    """Whether ``detect_batch`` preprocesses (normalize/scale/pad) ON the model
    device instead of on the host. Default ON — it removes the fixed host-numpy
    floor that otherwise dominates end-to-end time once the forward is on the GPU
    (Amdahl; see ``preprocess_torch``). ``PYXEM_NEURAL_GPU_PREP=0`` forces the host
    numpy path (the exact CPU reference) as an escape hatch. Irrelevant on CPU."""
    return os.environ.get("PYXEM_NEURAL_GPU_PREP", "").lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _build_input_stack(
    frames: np.ndarray,
    device,
    factor: float,
    bg_sigma: float,
    levels: int,
    auto_scale: bool,
    mad_floor: float | None = None,
):
    """Preprocess ``frames`` ((N,H,W)) into the padded ``(N,1,H',W')`` U-Net input
    tensor: scale → normalize → pad-to-multiple.

    On a GPU device (cuda/mps) with GPU-prep enabled this runs entirely ON the
    device via ``preprocess_torch`` (numerically matched to the numpy reference —
    see ``test_preprocess_torch_parity``), so nothing but the raw frames crosses
    the bus. On CPU (or if GPU-prep is disabled / a device op fails) it uses the
    canonical numpy/scipy path from ``preprocess.py`` and returns a CPU tensor.
    ``factor``/``bg_sigma`` are the shared per-batch knobs resolved by the caller."""
    if device.type in ("cuda", "mps") and _gpu_prep_enabled():
        try:
            from . import preprocess_torch as pt

            x = torch.from_numpy(frames).to(device)  # (N,H,W)
            if auto_scale and factor != 1.0:
                x = pt.scale_batch(x, factor)
            x = pt.normalize_input_batch(
                x, bg_sigma=bg_sigma, local=True, mad_floor=mad_floor
            )
            return pt.pad_to_multiple_batch(x, levels).unsqueeze(1)  # (N,1,H',W')
        except Exception as e:  # pragma: no cover — device-specific
            log.warning(
                "[models] on-device preprocessing failed (%s); falling back "
                "to host numpy preprocessing",
                e,
            )
    # Host numpy/scipy reference (CPU device, gpu-prep off, or a device-op failure).
    nrm_list = []
    for f in frames:
        if auto_scale and factor != 1.0:
            from scipy.ndimage import zoom

            f = zoom(f, factor, order=1)
        nrm_list.append(
            normalize_input(f, local=True, bg_sigma=bg_sigma, mad_floor=mad_floor)
        )
    stack = np.stack(nrm_list, 0)
    stack = np.stack([_pad_to_multiple(n, levels) for n in stack], 0)
    return torch.from_numpy(stack[:, None])  # CPU (N,1,H',W')


@torch.no_grad()
def detect_batch(
    model,
    frames,
    device,
    thresh: float = 0.3,
    min_distance: int = 4,
    auto_scale: bool = True,
    shared_scale: bool = True,
    bg_sigma: float | None = None,
    spot_diameter: float | None = None,
    with_width: bool = False,
):
    """Detect spots in a STACK of frames in one forward pass.

    ``with_width`` as in :func:`detect`; the decode is the same, so the batch and
    the single-frame preview give the same centers.

    ``frames`` is an (N,H,W) array (or a sequence of (H,W) arrays). Returns a list
    of N (Ni,3) [y,x,score] arrays in ORIGINAL frame coordinates — the same per-frame
    contract as ``detect``, but the model runs once on the whole batch and decoding
    stays GPU-resident (``center.decode_batch_centers``).

    All frames in a scan share the same physical disk size, so ``shared_scale``
    (default) estimates the scale factor ONCE from the first frame and applies it
    uniformly (cheaper, and what the scale-norm design recommends — "reuse one
    estimate across a stack"). The whole batch must share one H,W for a single
    tensor, so a shared factor is also required to keep frames the same shape.

    Memory: the padded stack is run through the U-Net in fixed sub-batches of
    ``PYXEM_NEURAL_BATCH`` frames (default 64), NOT as one (N,1,H,W) tensor — a
    single pass over a full nav chunk (1000+ frames) held ~16 GB of activations.
    The scale/NMS-window/bg_sigma are resolved ONCE from ``frames[0]`` (below,
    outside the sub-batch loop) so results are bit-identical to a single pass;
    only the forward+decode is chunked.

    Preprocessing (normalize/scale/pad) runs ON the model device when it's a GPU
    (``_build_input_stack`` → ``preprocess_torch``), so nothing but the raw frames
    crosses the bus; on CPU it uses the numpy/scipy reference. Both produce the same
    result (parity-tested), so switching devices never changes the peaks.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None]
    N = frames.shape[0]
    if N == 0:
        return []

    factor = 1.0
    if auto_scale:
        # One estimate for the whole stack (shared physical disk size); a known
        # ``spot_diameter`` (the user's Spot-size override) replaces the estimate.
        _ref, factor = scale_to_canonical(frames[0], diameter=spot_diameter)

    # Shared physical disk size across the scan -> resolve the size-dependent
    # NMS window + background sigma ONCE from the first frame (like the scale).
    md = max(2, int(round(min_distance * factor)))
    work_diam = _estimate_work_diam(frames[0], factor, spot_diameter)
    md, bg_sigma = _big_disk_params(md, bg_sigma, work_diam)
    radius = _width_radius(work_diam, md)

    levels = int(getattr(model, "levels", 2))
    # Padded (N,1,H',W') U-Net input — built on the model device (GPU) or on the
    # host (CPU); see _build_input_stack. Shared factor/bg_sigma keep every frame
    # one shape, so it's a single tensor sliced per sub-batch below.
    stack_t = _build_input_stack(
        frames,
        device,
        factor,
        bg_sigma,
        levels,
        auto_scale,
        mad_floor=getattr(model, "mad_floor", None),
    )
    mask_radius = _mask_radius(work_diam)

    K = _neural_sub_batch_size()
    per_frame: list = []
    # ``device`` / ``model`` may flip to CPU mid-call if an MPS op raises a
    # catchable RuntimeError (fix 2): re-move the model to CPU ONCE, retry the
    # failing sub-batch on CPU, and stay on CPU for the remaining sub-batches so
    # a flaky MPS op degrades cleanly at the finest grain instead of bubbling up
    # to the coarser ``_neural_block`` catch (which drops to a slow per-frame CPU
    # loop for the whole chunk). ``cur_*`` is the live model/device the loop uses.
    cur_model, cur_device = model, device
    for i0 in range(0, N, K):
        chunk = stack_t[i0 : i0 + K]
        # If even the CPU retry inside _forward_with_cpu_retry fails, the error
        # propagates to the caller's coarser fallback (per-frame CPU in
        # _neural_block) — no extra handling needed here.
        outputs, cur_model, cur_device = _forward_with_cpu_retry(
            cur_model, cur_device, chunk
        )
        res = decode_batch_centers(
            outputs[0],
            outputs[1],
            thresh=thresh,
            min_distance=md,
            radius=radius,
            mask_logits=outputs[2] if len(outputs) > 2 else None,
            mask_radius=mask_radius,
        )
        b = res[:, 0].long().cpu().numpy() if len(res) else np.zeros(0, np.int64)
        res_np = res.cpu().numpy()
        for f in range(chunk.shape[0]):
            per_frame.append(
                _frame_rows(res_np[b == f], frames.shape[1:], factor, with_width)
            )
        del outputs, res
        if cur_device.type == "cuda":
            torch.cuda.empty_cache()

    return per_frame


@torch.no_grad()
def calibrate(
    model,
    sample_frames,
    device,
    tune_threshold: bool = True,
    spot_diameter: float | None = None,
):
    """Calibration step (run ONCE on a few representative frames before a full scan).

    All frames in a 4D-STEM scan share the same physical disk size and background
    character, so we optimize the dataset-dependent knobs ONCE here and reuse them
    across the whole scan (cheap + hands-off). Optimizes:
      - ``bg_sigma``: confidence-maximizing local-norm high-pass scale (handles
        diffuse / beam-stop backgrounds without manual tuning).
      - ``scale_factor``: the disk-size -> canonical upscale (so tiny disks like
        SPED-Ag are seen at the trained size).
      - ``threshold``: lowered for faint-peak datasets where the confident response
        is weak (e.g. amorphous SRO / in-SEM), else the default 0.3.

    ``sample_frames``: a few (H,W) frames spread across the scan (ideally nav-blurred
    the same way the full compute will be). Returns a dict to pass into
    ``detect``/``detect_batch`` (bg_sigma, thresh) and ``scale_factor`` for reference.
    """
    frames = [np.asarray(f, np.float32) for f in sample_frames]
    if not frames:
        return {"bg_sigma": DEFAULT_BG_SIGMA, "thresh": 0.3, "scale_factor": 1.0}

    # 1) scale: one estimate (shared physical disk size). scale_to_canonical returns
    #    the upscale-only factor; a known spot_diameter overrides the estimate.
    _ref, factor = scale_to_canonical(frames[0], diameter=spot_diameter)
    work0 = _ref if factor != 1.0 else frames[0]

    # 2) bg_sigma: confidence-max over the sample (median best-sigma is robust).
    #    Big disks run near-native need bg ~ 0.6*diameter (a 12px high-pass carves
    #    the disk interior into a ring), so extend the candidate list with
    #    size-scaled values — the confidence-max picks them only when they win.
    candidates = [4, 5, 6, 8, 10, 12, 16, 20]
    wd = _estimate_work_diam(frames[0], factor, spot_diameter)
    if np.isfinite(wd) and wd > 20:
        candidates += [round(r * wd, 1) for r in (0.5, 0.6, 0.7)]
    sigmas = []
    confs = []
    for f in frames:
        w = f
        if factor != 1.0:
            from scipy.ndimage import zoom

            w = zoom(f, factor, order=1)
        s, c = auto_bg_sigma(model, w, device, candidates=tuple(candidates))
        sigmas.append(s)
        confs.append(c)
    bg = float(np.median(sigmas))
    best_conf = float(np.median(confs))

    # 3) threshold: if even the best-bg confidence is weak (faint-peak regime), lower
    #    the threshold so real faint peaks survive; otherwise keep the model's default.
    default = float(getattr(model, "default_threshold", 0.3))
    thresh = default
    if tune_threshold and best_conf < 0.45:
        thresh = float(np.clip(0.5 * best_conf, 0.1, default))

    return {
        "bg_sigma": bg,
        "thresh": thresh,
        "scale_factor": float(factor),
        "confidence": best_conf,
    }
