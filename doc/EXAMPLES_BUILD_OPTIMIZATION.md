# Examples Gallery Build Optimization

This document describes the optimization implemented for building the pyxem Examples Gallery.

## Problem

Several examples in the pyxem gallery are computationally intensive and/or require downloading large datasets from Zenodo:

- **Azimuthal Integration** (~53 seconds): Intensive calculations
- **Fluctuation Electron Microscopy (FEM)** (~53 seconds): Large data download + processing  
- **Glass Symmetry from Vectors** (~46 seconds): Large data download
- **Strain Mapping** (~30 seconds): Intensive simulated calculations

These slow examples significantly increase documentation build times, making CI builds and ReadtheDocs builds inefficient.

## Solution

An environment variable `PYXEM_SKIP_SLOW_EXAMPLES` has been implemented to conditionally skip these slow examples during automated documentation builds.

### How it Works

1. **Environment Variable**: Set `PYXEM_SKIP_SLOW_EXAMPLES=1` to skip slow examples
2. **Sphinx Configuration**: The `doc/conf.py` file checks this variable and updates the `ignore_pattern` for Sphinx Gallery accordingly
3. **CI Integration**: GitHub Actions and ReadtheDocs are configured to set this variable for faster builds
4. **Local Development**: Examples run normally unless the environment variable is explicitly set

### Usage

**Fast builds (skip slow examples):**
```bash
export PYXEM_SKIP_SLOW_EXAMPLES=1
make html
```

**Full builds (run all examples):**
```bash
unset PYXEM_SKIP_SLOW_EXAMPLES
# or
export PYXEM_SKIP_SLOW_EXAMPLES=0
make html
```

### Configuration Files

The following files have been updated to support this optimization:

- `doc/conf.py`: Core logic for conditional example execution
- `.github/workflows/docs.yml`: Set environment variable for CI builds
- `readthedocs.yaml`: Set environment variable for ReadtheDocs builds
- Example files: Added notes about potential skipping during automated builds

### Benefits

- **CI Builds**: Significantly faster documentation builds in GitHub Actions
- **ReadtheDocs**: Faster builds and reduced resource usage
- **Local Development**: Full flexibility - run fast or complete builds as needed
- **Backwards Compatible**: No changes to existing behavior unless environment variable is set

### Future Enhancements

- Data caching could be added to speed up examples that download datasets
- Additional examples could be marked as slow/fast for more granular control
- Build artifacts could be cached between documentation builds