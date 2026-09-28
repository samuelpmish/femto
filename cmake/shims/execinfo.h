#pragma once

// emscripten's sysroot has no <execinfo.h>, but cuBQL includes it whenever
// __GNUC__ is defined (which clang/emscripten does). This shim is only on the
// include path for webassembly builds (see the EMSCRIPTEN branch in the
// top-level CMakeLists.txt), and stubs out the backtrace functions
// cuBQL uses for its error reporting.

#include <cstddef>

inline int backtrace(void **, int) { return 0; }
inline char ** backtrace_symbols(void * const *, int) { return nullptr; }
inline void backtrace_symbols_fd(void * const *, int, int) {}
