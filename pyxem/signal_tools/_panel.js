// The find-vectors parameter panel (see _panel.py). Plain DOM, no build step.

const SLIDERS = [
  // [trait, label, min, max, step, decimals, tooltip]
  ["threshold", "Threshold", 0, 1, 0.01, 2, "The network's confidence a spot needs"],
  ["min_distance", "Min distance", 1, 30, 1, 0, "Minimum spacing between spots (px)"],
  ["spot_radius", "Spot size", 0, 30, 0.5, 1, "Spot radius (px); 0 estimates it from the pattern"],
  ["navigation_sigma", "Nav blur σ", 0, 5, 0.1, 1, "Gaussian blur over neighboring scan positions"],
];
const CHECKS = [
  ["refine_centers", "Refine centers", "Re-place each disk with the center network"],
  ["friedel_partner", "Friedel partner", "Also refine with each disk's Friedel-mirror window"],
  ["neighbor_refine", "Neighbor refine", "Compute only: keep spots that recur in neighboring scan positions"],
];

function element(tag, className, attributes = {}) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function isLight(node) {
  const body = document.body;
  const jupyter = body.getAttribute("data-jp-theme-light");
  if (jupyter === "true") return true;
  if (jupyter === "false") return false;
  if (body.classList.contains("vscode-light")) return true;
  if (body.classList.contains("vscode-dark") || body.classList.contains("vscode-high-contrast")) return false;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
}

function slider(model, [name, label, min, max, step, decimals, tooltip]) {
  const cell = element("div", "pxm-fv-cell", { title: tooltip, "data-control": name });
  const text = element("span", "pxm-fv-cell-label");
  text.textContent = label;
  const range = element("input", "", { type: "range", min, max, step });
  const number = element("input", "pxm-fv-number", { type: "number", min, max, step });
  const show = () => {
    const value = model.get(name);
    range.value = value;
    if (document.activeElement !== number) number.value = Number(value).toFixed(decimals);
  };
  const send = (value) => {
    if (!Number.isFinite(value)) return;
    model.set(name, Math.min(max, Math.max(min, value)));
    model.save_changes();
  };
  range.addEventListener("input", () => send(Number(range.value)));
  number.addEventListener("change", () => send(Number(number.value)));
  number.addEventListener("blur", show);
  model.on(`change:${name}`, show);
  show();
  cell.append(text, range, number);
  return cell;
}

function check(model, [name, label, tooltip]) {
  const wrapper = element("label", "pxm-fv-check", { title: tooltip, "data-control": name });
  const box = element("input", "", { type: "checkbox" });
  box.addEventListener("change", () => {
    model.set(name, box.checked);
    model.save_changes();
  });
  const show = () => { box.checked = model.get(name); };
  model.on(`change:${name}`, show);
  show();
  wrapper.append(box, document.createTextNode(label));
  return wrapper;
}

function toggle(model, name, label, tooltip) {
  const button = element("button", "pxm-fv-toggle", { type: "button", role: "switch", title: tooltip, "data-control": name });
  const track = element("span", "pxm-fv-track");
  track.append(element("span", "pxm-fv-knob"));
  button.append(track, document.createTextNode(label));
  const show = () => button.setAttribute("aria-checked", String(model.get(name)));
  button.addEventListener("click", () => {
    model.set(name, !model.get(name));
    model.save_changes();
  });
  model.on(`change:${name}`, show);
  show();
  return button;
}

function render({ model, el }) {
  const root = element("div", "pxm-fv");
  const applyTheme = () => root.classList.toggle("pxm-fv--light", isLight(root));
  applyTheme();
  const observer = new MutationObserver(applyTheme);
  observer.observe(document.body, { attributes: true, attributeFilter: ["class", "data-jp-theme-light"] });
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
  if (media && media.addEventListener) media.addEventListener("change", applyTheme);

  // header
  const head = element("div", "pxm-fv-head");
  const title = element("span", "pxm-fv-title");
  title.textContent = model.get("title");
  const close = element("button", "pxm-fv-x", { title: "Close", "data-control": "close" });
  close.textContent = "✕";
  close.addEventListener("click", () => {
    model.set("close_clicks", model.get("close_clicks") + 1);
    model.save_changes();
  });
  head.append(title, close);
  const hint = element("div", "pxm-fv-hint");
  hint.textContent = model.get("hint");

  // model + response toggle
  const modelRow = element("div", "pxm-fv-row");
  const modelLabel = element("label", "pxm-fv-row-label");
  modelLabel.textContent = "Model";
  const select = element("select", "pxm-fv-select", { "data-control": "model" });
  const fillModels = () => {
    select.replaceChildren(...model.get("models").map((name) => {
      const option = element("option", "", { value: name });
      option.textContent = name;
      return option;
    }));
    select.value = model.get("model");
  };
  select.addEventListener("change", () => {
    model.set("model", select.value);
    model.save_changes();
  });
  model.on("change:models", fillModels);
  model.on("change:model", () => { select.value = model.get("model"); });
  fillModels();
  const grow = element("div", "pxm-fv-grow");
  grow.append(select);
  modelRow.append(modelLabel, grow,
    toggle(model, "show_response", "Heatmap", "Show where the network fired"));

  // sliders, two per row
  const sliders = element("div", "pxm-fv-grid");
  sliders.append(...SLIDERS.map((spec) => slider(model, spec)));

  // center steps
  const checks = element("div", "pxm-fv-grid");
  const checkNodes = CHECKS.map((spec) => check(model, spec));
  checks.append(...checkNodes);
  const neighbor = checkNodes[2];
  const showNeighbor = () => {
    const available = model.get("neighbor_refine_available");
    neighbor.classList.toggle("pxm-fv-check--off", !available);
    neighbor.querySelector("input").disabled = !available;
  };
  model.on("change:neighbor_refine_available", showNeighbor);
  showNeighbor();

  // Extra: the beam stop
  const disclosure = element("div", "pxm-fv-disclosure");
  const disclosureButton = element("button", "pxm-fv-disclosure-button", { type: "button", "data-control": "extra" });
  const chevron = element("span", "pxm-fv-chevron");
  chevron.textContent = "›";
  disclosureButton.append(chevron, document.createTextNode("Extra"));
  const page = element("div", "pxm-fv-page");
  const dilation = slider(model, ["mask_dilation", "Dilate", 0, 20, 1, 0, "Pixels the beam-stop mask is grown by"]);
  const beamRow = element("div", "pxm-fv-grid");
  beamRow.append(check(model, ["mask_beam_stop", "Mask beam stop", "Report no spot inside the beam stop"]), dilation);
  page.append(beamRow);
  disclosure.append(disclosureButton, page);
  const showDisclosure = () => {
    disclosure.setAttribute("data-open", String(model.get("extra_open")));
    dilation.style.visibility = model.get("mask_beam_stop") ? "visible" : "hidden";
  };
  disclosureButton.addEventListener("click", () => {
    model.set("extra_open", !model.get("extra_open"));
    model.save_changes();
  });
  model.on("change:extra_open", showDisclosure);
  model.on("change:mask_beam_stop", showDisclosure);
  showDisclosure();

  // actions and status
  const actions = element("div", "pxm-fv-actions");
  const compute = element("button", "pxm-fv-primary", { "data-control": "compute", title: "Find the vectors in every pattern" });
  const closeButton = element("button", "pxm-fv-secondary", { "data-control": "close-button" });
  closeButton.textContent = "Close";
  closeButton.addEventListener("click", () => close.click());
  const showBusy = () => {
    compute.disabled = model.get("busy");
    compute.textContent = model.get("busy") ? "Running…" : "Compute";
  };
  compute.addEventListener("click", () => {
    model.set("compute_clicks", model.get("compute_clicks") + 1);
    model.save_changes();
  });
  model.on("change:busy", showBusy);
  showBusy();
  actions.append(compute, closeButton);
  const status = element("div", "pxm-fv-status", { "data-control": "status" });
  const showStatus = () => { status.innerHTML = model.get("status"); };
  model.on("change:status", showStatus);
  showStatus();

  root.append(head, hint, modelRow, sliders, checks, disclosure, actions, status);
  el.append(root);
  return () => {
    observer.disconnect();
    if (media && media.removeEventListener) media.removeEventListener("change", applyTheme);
  };
}

export default { render };
