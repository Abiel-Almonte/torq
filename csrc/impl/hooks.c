#include "../hooks.h"

typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);

#define _CHECK(res, message) { \
    if (!res) { \
        fprintf(stderr, message); \
        exit(EXIT_FAILURE); \
    } \
} \

static _Bool is_synchronization_hooked = 0;
PyObject* synchronization_hook(PyObject* self, PyObject* args){
    int value;

    if (!PyArg_ParseTuple(args, "p", &value)){
        return NULL;
    }

    is_synchronization_hooked = (_Bool)value;

    Py_RETURN_NONE;
}


cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static void* handler = NULL;
    static cudaStreamSynchronize_fn real_synchronize = NULL;

    if (!real_synchronize) {
        handler = dlopen("/usr/lib/libcudart.so", RTLD_LAZY);
        
        if (!handler) {
            fprintf(stderr, "%s\n", dlerror());
            exit(EXIT_FAILURE);
        }
        
        real_synchronize = (cudaStreamSynchronize_fn)dlsym(handler, "cudaStreamSynchronize");
        _CHECK(real_synchronize, "Unable to fetch real cudaStreamSynchronize.");
    }   

    if (is_synchronization_hooked) {
        printf("sync called\n");
    }

    return real_synchronize(stream);
}
