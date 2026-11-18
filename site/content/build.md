---
title: Building
---

# Building

femto builds with CMake (3.24 or newer) and a C++20 compiler. Third-party
dependencies are fetched and built by CMake itself.

On Ubuntu 24.04, if you enable graphics, install a few packages first:

```sh
sudo apt install libglu1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

## Native build

```sh
cmake -S . -B build -GNinja -C options.cmake
cmake --build build
```

[options.cmake](https://github.com/samuelpmish/femto/blob/main/options.cmake)
holds the defaults; any option can also be given as `-DNAME=ON|OFF`.

## Build options

| Option | Meaning |
|---|---|
| `FEMTO_ENABLE_EXAMPLES` | build the native programs in `examples/cpp` |
| `FEMTO_ENABLE_TESTING` | build the unit tests |
| `FEMTO_ENABLE_TOOLS` | build the helper tools in `tools/` |
| `FEMTO_ENABLE_GRAPHICS` | build the OpenGL viewer |
| `FEMTO_ENABLE_OPENMP` | use OpenMP for threading |
| `FEMTO_ENABLE_CUDA` | build the GPU backend (needs cuDSS) |
| `FEMTO_ENABLE_MKL` / `MKL_ROOT` | use Intel MKL's sparse direct solver |
| `FEMTO_ENABLE_PASTIX` | build the PaStiX solver from source (the default whenever MKL is off) |
| `FEMTO_ENABLE_PCH` | precompiled headers |
| `FEMTO_ENABLE_ASAN` | address sanitizer |
| `FEMTO_ENABLE_JIT`, `FEMTO_ENABLE_ENZYME`, `FEMTO_ENABLE_MATHEMATICA` | experimental code generation |
| `FEMTO_USE_MOLD` | link with mold |
| `FEMTO_WASM_THREADS` | WebAssembly only: pthread support (default `ON`) |

## WebAssembly

The browser examples are built with [emscripten](https://emscripten.org):

```sh
source ~/emsdk/emsdk_env.sh
emcmake cmake -B build-web-mt -DCMAKE_BUILD_TYPE=Release
cmake --build build-web-mt --target femto_examples_html
cd build-web-mt/examples/html && python3 serve.py 8000
```

The threaded module needs a cross-origin-isolated page. `serve.py` sends the
required headers locally; on GitHub Pages a service worker
(`enable-threads.js`) supplies them. Configure with `-DFEMTO_WASM_THREADS=OFF`
for a single-threaded module that any file server can host.
