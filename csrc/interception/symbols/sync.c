#include "sync.h"
#include "../state.h"


cudaError_t cudaDeviceSynchronize(void){
    TRACE();
    static cudaDeviceSynchronize_fn real_synchronize = NULL;

    _LOAD_CUDART_SYMBOL(cudaDeviceSynchronize, real_synchronize);

    if (sync_detection_enabled) {
        sync_detected = true;
    }

    return real_synchronize();
}


cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    TRACE();
    static cudaStreamSynchronize_fn real_synchronize = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamSynchronize, real_synchronize);

    if (sync_detection_enabled) {
        sync_detected = true;
    }

    return real_synchronize(injected_stream? (*injected_stream) : stream);
}

cudaError_t cudaGetDevice(int* device){
    TRACE();

    static cudaGetDevice_fn real_getdevice = NULL;

    _LOAD_CUDART_SYMBOL(cudaGetDevice, real_getdevice);


    return real_getdevice(device);
}
