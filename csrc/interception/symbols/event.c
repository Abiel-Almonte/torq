#include "event.h"
#include "../state.h"

cudaError_t cudaEventRecord(
    cudaEvent_t event,
    cudaStream_t stream
) {
    TRACE();
    cudaEventRecord_fn real_event_record = NULL;
    
    _LOAD_CUDART_SYMBOL(cudaEventRecord, real_event_record);

    return real_event_record(
        event,
        (injected_stream) ? *injected_stream : stream
    );
}