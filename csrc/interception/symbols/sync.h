#ifndef _TORQ_SYNC_H
#define _TORQ_SYNC_H

#include "../utils.h"


// cudaStreamSynchronize (Runtime API)
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);

// cudaDeviceSynchronize (Runtime API)
cudaError_t cudaDeviceSynchronize(void);
typedef cudaError_t (*cudaDeviceSynchronize_fn)(void);

// cudaGetDevice (Runtime API)
cudaError_t cudaGetDevice(int*);
typedef cudaError_t (*cudaGetDevice_fn)(int*);

#endif
