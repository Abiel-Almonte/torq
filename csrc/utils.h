#ifndef _TORQ_UTILS_H
#define _TORQ_UTILS_H

#include <Python.h>
#include <cuda_runtime.h>


#define _CUDA_CHECK(expr, message) { \
    cudaError_t _err = expr; \
    if (_err != cudaSuccess) { \
        PyErr_SetString(PyExc_RuntimeError, (message)); \
        return NULL; \
    } \
} \

#endif