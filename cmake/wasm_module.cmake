# link options for an embind module: one self-contained .js file exporting a
# factory function.  Used by examples/html/ (the physics examples), configured
# through emscripten's emcmake wrapper

function(femto_wasm_module target export_name)
  target_link_libraries(${target} PRIVATE femto)
  target_link_options(${target} PRIVATE
    -lembind
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=${export_name}"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sENVIRONMENT=web,node,worker"  # node is included so the module can be smoke-tested headlessly
    "-sFORCE_FILESYSTEM=1"           # the examples read meshes from it
    "-sEXPORTED_RUNTIME_METHODS=FS"  # ... which the page writes there after fetching them
    "-sSTACK_SIZE=8388608"
  )
  if (FEMTO_WASM_THREADS)
    # wasm cannot spawn threads on demand from compiled code, so the worker
    # pool must be preallocated at startup. It backs femto's thread pool plus
    # the PaStiX solver threads (each live factorization keeps its own crew,
    # and the analyses hold up to two at a time), hence 3x hardwareConcurrency.
    target_link_options(${target} PRIVATE
      "-sPTHREAD_POOL_SIZE=3*navigator.hardwareConcurrency"
      "-sDEFAULT_PTHREAD_STACK_SIZE=2097152"
      # shared-memory growth stalls every thread while it happens, so start
      # with a heap large enough for the analyses instead of growing into it
      "-sINITIAL_MEMORY=1073741824"
      "-sMAXIMUM_MEMORY=4294967296"
      "-sMALLOC=mimalloc"   # emscripten's default dlmalloc serializes on a global lock
      "-Wno-pthreads-mem-growth"
    )
  endif()
endfunction()
