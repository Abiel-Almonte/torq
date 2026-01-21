#ifndef _TORQ_UTILS_H
#define _TORQ_UTILS_H
#include <Python.h>
#include <cuda.h> // cuda driver api
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

#endif