#ifndef _TORQ_UTILS_H
#define _TORQ_UTILS_H

#include <Python.h>
#include <cuda_runtime.h>


// call python function from c functions
#define _WITH_PYTHON_GIL(expr) { \
    PyGILState_STATE gstate; \
    gstate = PyGILState_Ensure(); \
    expr \
    PyGILState_Release(gstate); \
}

/// CUDA MACROS ///

#define _CUDA_CHECK(expr, message) { \
    cudaError_t _err = expr; \
    if (_err != cudaSuccess) { \
        PyErr_SetString(PyExc_RuntimeError, (message)); \
        return NULL; \
    } \
} \

// get the real symbol from cuda runtime
// TODO static dispatch for windows
#define _LOAD_CUDART_SYMBOL(real_fn_symbol, real_fn) { \
    if (!(real_fn)) { \
        handler = dlopen("libcudart.so", RTLD_LAZY); \
        if (!(handler)) { \
            fprintf(stderr, "dlopen failed: %s\n", dlerror()); \
            exit(EXIT_FAILURE); \
        } \
        real_fn = (real_fn_symbol##_fn)dlsym(handler, #real_fn_symbol); \
        if (!real_fn) { \
            fprintf(stderr, "Unable to fetch real " #real_fn_symbol); \
            exit(EXIT_FAILURE); \
        } \
    } \
}


#endif