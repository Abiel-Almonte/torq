#ifndef _TORQ_HOOKS_H
#define _TORQ_HOOKS_H
#include "utils.h"
#include <dlfcn.h> // load real symbols from dynamic linked library

// toggles
PyObject* synchronization_hook(PyObject* self, PyObject* args);

// overrided symbols
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

#endif

