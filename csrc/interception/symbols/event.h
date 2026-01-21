#ifndef _TORQ_EVENT_H
#define _TORQ_EVENT_H
#include "../utils.h"

// cudaEventRecord (Runtime API)
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
typedef cudaError_t (*cudaEventRecord_fn)(cudaEvent_t, cudaStream_t);

#endif