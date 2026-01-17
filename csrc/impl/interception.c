#include "../interception.h"

static _Bool sync_detected = 0;
PyObject* get_sync_detected(PyObject* self, PyObject* args){
    if (sync_detected == 1){
        Py_RETURN_TRUE;
    } 

    Py_RETURN_FALSE;
}

PyObject* clear_sync_detected(PyObject* self, PyObject* args){
    sync_detected = 0;
    Py_RETURN_NONE;
}

static _Bool sync_detection_enabled = 0;
PyObject* detect_sync(PyObject* self, PyObject* args){
    int value;

    if (!PyArg_ParseTuple(args, "p", &value)){
        return NULL;
    }

    sync_detection_enabled = (_Bool)value;

    Py_RETURN_NONE;
}

static _Bool kernel_detected = 0;
PyObject* get_kernel_detected(PyObject* self, PyObject* args){
    if (kernel_detected == 1){
        Py_RETURN_TRUE;
    } 

    Py_RETURN_FALSE;
}

PyObject* clear_kernel_detected(PyObject* self, PyObject* args){
    kernel_detected = 0;
    Py_RETURN_NONE;
}

static _Bool kernel_detection_enabled = 0;
PyObject* detect_kernel(PyObject* self, PyObject* args){
    int value;

    if (!PyArg_ParseTuple(args, "p", &value)){
        return NULL;
    }

    kernel_detection_enabled = (_Bool)value;

    Py_RETURN_NONE;
}


cudaError_t cudaDeviceSynchronize(void){
    static void* handler = NULL;
    static cudaDeviceSynchronize_fn real_synchronize = NULL;

    _LOAD_CUDART_SYMBOL(cudaDeviceSynchronize, real_synchronize);

    if (sync_detection_enabled) {
        sync_detected = 1;
    }

    return real_synchronize();
}


cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static void* handler = NULL;
    static cudaStreamSynchronize_fn real_synchronize = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamSynchronize, real_synchronize);

    if (sync_detection_enabled) {
        sync_detected = 1;
    }

    return real_synchronize(stream);
}

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
) {
    static void* handler = NULL;
    static cuLaunchKernel_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(cuLaunchKernel, real_kernel_launch);

    if (kernel_detection_enabled){
        kernel_detected = 1;
    }

    return real_kernel_launch(
        f, 
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        hStream,
        kernelParams,
        extra
    );
}

CUresult cuLaunchKernelEx (
    const CUlaunchConfig* config,
    CUfunction f,
    void** kernelParams,
    void** extra
) {
    static void* handler = NULL;
    static cuLaunchKernelEx_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(cuLaunchKernelEx, real_kernel_launch);

    if (kernel_detection_enabled){
        kernel_detected = 1;
    }

    return real_kernel_launch(
        config,
        f,
        kernelParams,
        extra
    );
}

cudaError_t cudaLaunchKernel ( 
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream
) {
    static void* handler = NULL;
    static cudaLaunchKernel_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(cudaLaunchKernel, real_kernel_launch);

    if (kernel_detection_enabled){
        kernel_detected = 1;
    }

    return real_kernel_launch(
        func,
        gridDim, blockDim,
        args,
        sharedMem,
        stream
    );
}

cudaError_t cudaLaunchKernelExc (
    const cudaLaunchConfig_t* config,
    const void* func,
    void** args
) {
    static void* handler = NULL;
    static cudaLaunchKernelExc_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(cudaLaunchKernelExc, real_kernel_launch);

    if (kernel_detection_enabled){
        kernel_detected = 1;
    }

    return real_kernel_launch(
        config,
        func,
        args
    );
}