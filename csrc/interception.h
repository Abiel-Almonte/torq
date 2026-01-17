#ifndef _TORQ_INTERCEPTION_H
#define _TORQ_INTERCEPTION_H
#include "utils.h"
#include <dlfcn.h> // to load real symbols from dynamic linked library
#include "cuda.h" // CUfunction & CUstream

// sync detection
PyObject* detect_sync(PyObject* self, PyObject* args);
PyObject* clear_sync_detected(PyObject* self, PyObject* args);
PyObject* get_sync_detected(PyObject* self, PyObject* args);

// kernel launch detection
PyObject* detect_kernel(PyObject* self, PyObject* args);
PyObject* clear_kernel_detected(PyObject* self, PyObject* args);
PyObject* get_kernel_detected(PyObject* self, PyObject* args);

// override cudaStreamSynchronize symbol
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);

// override cudaDeviceSynchronize symbol
cudaError_t cudaDeviceSynchronize(void);
typedef cudaError_t (*cudaDeviceSynchronize_fn)(void);

// override cuLaunchKernel symbol
CUresult cuLaunchKernel (
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
);
typedef CUresult (*cuLaunchKernel_fn)(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
);

// override cuLaunchKernelEx symbol
CUresult cuLaunchKernelEx (
    const CUlaunchConfig* config,
    CUfunction f,
    void** kernelParams,
    void** extra
);
typedef CUresult (*cuLaunchKernelEx_fn)(
    const CUlaunchConfig* config,
    CUfunction f,
    void** kernelParams,
    void** extra
);

// override cudaLaunchKernel symbol
cudaError_t cudaLaunchKernel ( 
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream
);
typedef cudaError_t (*cudaLaunchKernel_fn) ( 
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream
);

// override cudaLaunchKernelExc symbol
cudaError_t cudaLaunchKernelExc (
    const cudaLaunchConfig_t* config,
    const void* func,
    void** args
);
typedef cudaError_t (*cudaLaunchKernelExc_fn) (
    const cudaLaunchConfig_t* config,
    const void* func,
    void** args
);

#endif

