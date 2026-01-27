#include "memory.h"
#include "../tracer.h"
#include "../state.h"


CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    TRACE();
    static cuMemAlloc_v2_fn real_alloc = NULL;

    _LOAD_CUDA_SYMBOL(cuMemAlloc_v2, real_alloc);

    if (global_tracer) {
        if (global_tracer->status.done){
            *dptr = (CUdeviceptr)next_device_ptr(global_tracer);
            return CUDA_SUCCESS;
        }

        CUresult err = real_alloc(dptr, bytesize);

        if (err == CUDA_SUCCESS && global_tracer->status.in_progress) {
            tracer_push(global_tracer, (void*)*dptr);
        }
        
        return err;
    }

    return real_alloc(dptr, bytesize);
}

CUresult cuMemcpyAsync (
    CUdeviceptr dst,
    CUdeviceptr src, 
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyAsync_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyAsync, real_memcpy);


    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, (void*)dst);
        } else if (global_tracer->status.done) {
            dst = (CUdeviceptr)next_device_ptr(global_tracer);
        }
    }

    return real_memcpy(
        dst,
        src,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemsetD8Async (
    CUdeviceptr dst,
    unsigned char uc,
    size_t N,
    CUstream hStream
) {
    TRACE();
    static cuMemsetD8Async_fn real_memset = NULL;

    _LOAD_CUDA_SYMBOL(cuMemsetD8Async, real_memset);

    if (global_tracer) {
        if (global_tracer->status.in_progress){
            tracer_push(global_tracer, (void*)dst);
        } else if (global_tracer->status.done) {
            dst = (CUdeviceptr)next_device_ptr(global_tracer);
        }
    }

    return real_memset(
        dst,
        uc,
        N,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemsetD32Async (
    CUdeviceptr dst,
    unsigned int ui,
    size_t N,
    CUstream hStream
) {
    TRACE();
    static cuMemsetD32Async_fn real_memset = NULL;

    _LOAD_CUDA_SYMBOL(cuMemsetD32Async, real_memset);

    if (global_tracer) {
        if (global_tracer->status.in_progress){
            tracer_push(global_tracer, (void*)dst);
        } else if (global_tracer->status.done) {
            dst = (CUdeviceptr)next_device_ptr(global_tracer);
        }
    }

    return real_memset(
        dst,
        ui,
        N,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpy2DAsync_v2(
    const CUDA_MEMCPY2D* pCopy,
    CUstream hStream
){
    TRACE();
    static cuMemcpy2DAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpy2DAsync_v2, real_memcpy);

    if (global_tracer) {
        if (global_tracer->status.in_progress){
            tracer_push(global_tracer, (void*)pCopy->dstDevice);
        } else if (global_tracer->status.done) {
            CUdeviceptr new_dstDevice = {(size_t)next_device_ptr(global_tracer)};
            const CUDA_MEMCPY2D new_pCopy = {
                pCopy->srcXInBytes,
                pCopy->srcY,
                pCopy->srcMemoryType,
                pCopy->srcHost,
                pCopy->srcDevice,
                pCopy->srcArray,
                pCopy->srcPitch,
                pCopy->dstXInBytes,
                pCopy->dstY,
                pCopy->dstMemoryType,
                pCopy->dstHost,
                new_dstDevice,
                pCopy->dstArray,
                pCopy->dstPitch,
                pCopy->WidthInBytes,
                pCopy->Height
            };
            
            return real_memcpy(
                &new_pCopy,
                (injected_stream)? (CUstream)(*injected_stream) : hStream
            );
        }
    }

    return real_memcpy(
        pCopy,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpy3DAsync_v2(
    const CUDA_MEMCPY3D* pCopy,
    CUstream hStream
){
    TRACE();
    static cuMemcpy3DAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpy3DAsync_v2, real_memcpy);

    if (global_tracer) { 
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, (void*)pCopy->dstDevice);
        } else if (global_tracer->status.done) {
            CUdeviceptr new_dstDevice = {(size_t)next_device_ptr(global_tracer)};
            const CUDA_MEMCPY3D new_pCopy = {
                pCopy->srcXInBytes,
                pCopy->srcY,
                pCopy->srcZ,
                pCopy->srcLOD,
                pCopy->srcMemoryType,
                pCopy->srcHost,
                pCopy->srcDevice,
                pCopy->srcArray,
                pCopy->reserved0,
                pCopy->srcPitch,
                pCopy->srcHeight,
                pCopy->dstXInBytes,
                pCopy->dstY,
                pCopy->dstZ,
                pCopy->dstLOD,
                pCopy->dstMemoryType,
                pCopy->dstHost,
                new_dstDevice,
                pCopy->dstArray,
                pCopy->reserved1,
                pCopy->dstPitch,
                pCopy->dstHeight,
                pCopy->WidthInBytes,
                pCopy->Height,
                pCopy->Depth
            };
        
            return real_memcpy(
                &new_pCopy,
                (injected_stream)? (CUstream)(*injected_stream) : hStream
            );
        }
    }

    return real_memcpy(
        pCopy,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpyHtoDAsync_v2 (
    CUdeviceptr dst,
    const void* src, 
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyHtoDAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyHtoDAsync_v2, real_memcpy);

    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, (void*)dst);
        } else if (global_tracer->status.done) {
            dst = (CUdeviceptr)next_device_ptr(global_tracer);
        }
    }

    return real_memcpy(
        dst,
        src,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpyDtoHAsync_v2 (
    void* dstHost, 
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyDtoHAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyDtoHAsync_v2, real_memcpy);

    return real_memcpy(
        dstHost,
        srcDevice,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpyDtoDAsync_v2 (
    CUdeviceptr dstDevice, 
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyDtoDAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyDtoDAsync_v2, real_memcpy);

    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, (void*)dstDevice);
        } else if(global_tracer->status.done) {
            dstDevice = (CUdeviceptr)next_device_ptr(global_tracer);
        }
    }

    return real_memcpy(
        dstDevice,
        srcDevice,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    TRACE();
    static cudaMalloc_fn real_malloc = NULL;

    _LOAD_CUDART_SYMBOL(cudaMalloc, real_malloc);

    if (global_tracer) {
        if (global_tracer->status.done) {
            *devPtr = next_device_ptr(global_tracer);
            return cudaSuccess;
        }

        cudaError_t err = real_malloc(devPtr, size);
    
        if (err == cudaSuccess && global_tracer->status.in_progress) {
            tracer_push(global_tracer, *devPtr);
        }

        return err;

    }
    
    return real_malloc(devPtr, size);
}

cudaError_t cudaMemcpyAsync (
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
) { 
    TRACE();
    static cudaMemcpyAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemcpyAsync, real_memcpy);
    
    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, dst);
        }  else if (global_tracer->status.done) {
            dst = next_device_ptr(global_tracer);
        }
    }

    return real_memcpy(
        dst,
        src,
        count,
        kind,
        (injected_stream)? *injected_stream : stream
    );
}

cudaError_t cudaMemcpy2DAsync (
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
) { 
    TRACE();
    static cudaMemcpy2DAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemcpy2DAsync, real_memcpy);
        
    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, dst);
        }  else if (global_tracer->status.done) {
            dst = next_device_ptr(global_tracer);
        }
    }

    return real_memcpy(
        dst,
        dpitch,
        src,
        spitch,
        width,
        height,
        kind,
        (injected_stream)? *injected_stream : stream
    );
}

cudaError_t cudaMemcpy3DAsync(
    const struct cudaMemcpy3DParms* p,
    cudaStream_t stream
){
    TRACE();
    static cudaMemcpy3DAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemcpy3DAsync, real_memcpy);

    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, ((void*)p->dstPtr.ptr));
        }  else if (global_tracer->status.done) {
            struct cudaPitchedPtr new_dstPtr = {
                next_device_ptr(global_tracer), 
                p->dstPtr.pitch, 
                p->dstPtr.xsize, 
                p->dstPtr.ysize
            };
            const struct cudaMemcpy3DParms new_p = {
                p->srcArray,
                p->srcPos,
                p->srcPtr,
                p->dstArray,
                p->dstPos,
                p->dstPtr, // TODO figure out if i need to replace dst ptr and dst array
                p->extent,
                p->kind
            };

            return real_memcpy(
                &new_p,
                (injected_stream)? *injected_stream : stream
            );
        }
    }

    return real_memcpy(
        p,
        (injected_stream)? *injected_stream : stream
    );
} 

cudaError_t cudaMemsetAsync(
    void* devPtr,
    int  value,
    size_t count,
    cudaStream_t stream
) {
    TRACE();
    static cudaMemsetAsync_fn real_memset = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemsetAsync, real_memset);

    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, devPtr);
        } else if (global_tracer->status.done) {
            devPtr = next_device_ptr(global_tracer);
        }
    }

    return real_memset( 
        devPtr,
        value,
        count,
        (injected_stream)? (*injected_stream) : stream
    );
}

cudaError_t cudaMemset2DAsync  (
    void* devPtr,
    size_t pitch,
    int value,
    size_t width,
    size_t height,
    cudaStream_t stream 
) {
    TRACE();
    static cudaMemset2DAsync_fn real_memset = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemset2DAsync, real_memset);

    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, devPtr);
        } else if (global_tracer->status.done) {
            devPtr = next_device_ptr(global_tracer);
        }
    }

    return real_memset( 
        devPtr,
        pitch,
        value,
        width,
        height,
        (injected_stream)? (*injected_stream) : stream
    );
}

cudaError_t cudaMemset3DAsync  (
    struct cudaPitchedPtr pitchedDevPtr,
    int value,
    struct cudaExtent extent,
    cudaStream_t stream
) {
    TRACE();
    static cudaMemset3DAsync_fn real_memset = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemset3DAsync, real_memset);

    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, (void*)pitchedDevPtr.ptr);
        } else if (global_tracer->status.done) {
            struct cudaPitchedPtr new_pitchedDevPtr = {
                next_device_ptr(global_tracer), 
                pitchedDevPtr.pitch, 
                pitchedDevPtr.xsize, 
                pitchedDevPtr.ysize
            };

            return real_memset( 
                new_pitchedDevPtr,
                value,
                extent,
                (injected_stream)? (*injected_stream) : stream
            );
        }
    }

    return real_memset( 
        pitchedDevPtr,
        value,
        extent,
        (injected_stream)? (*injected_stream) : stream
    );
}


cudaError_t cudaStreamAttachMemAsync (
    cudaStream_t stream,
    void* devPtr,
    size_t length,
    unsigned int flag
) {
    TRACE();
    static cudaStreamAttachMemAsync_fn real_attach = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamAttachMemAsync, real_attach);

    if (global_tracer) {
        if (global_tracer->status.in_progress) {
            tracer_push(global_tracer, devPtr);
        } else if (global_tracer->status.done) {
            devPtr = next_device_ptr(global_tracer);
        }
    }

    return real_attach(
        (injected_stream)? *injected_stream : stream,
        devPtr,
        length,
        flag
    );
}
