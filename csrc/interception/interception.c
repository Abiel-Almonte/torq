#include "interception.h"
#include "state.h"


void* cudart_handler = NULL;
void* cuda_handler = NULL;
void* cudnn_handler = NULL;

bool sync_detection_enabled = false;
bool sync_detected = false;

bool kernel_detection_enabled = false;
bool kernel_detected = false;

cudaStream_t* injected_stream = NULL;


PyObject* get_sync_detected(PyObject* self, PyObject* args){
    if (sync_detected) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* clear_sync_detected(PyObject* self, PyObject* args){
    sync_detected = false;
    Py_RETURN_NONE;
}

PyObject* detect_sync(PyObject* self, PyObject* args){
    int value;

    if (!PyArg_ParseTuple(args, "p", &value)){
        return NULL;
    }

    sync_detection_enabled = (bool)value;

    Py_RETURN_NONE;
}

PyObject* get_kernel_detected(PyObject* self, PyObject* args){
    if (kernel_detected) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* clear_kernel_detected(PyObject* self, PyObject* args){
    kernel_detected = false;
    Py_RETURN_NONE;
}

PyObject* detect_kernel(PyObject* self, PyObject* args){
    int value;

    if (!PyArg_ParseTuple(args, "p", &value)){
        return NULL;
    }

    kernel_detection_enabled = (bool)value;

    Py_RETURN_NONE;
}


PyObject* inject_stream(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)){
        return NULL;
    }

    injected_stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t");

    Py_RETURN_NONE;
}

PyObject* clear_injection(PyObject* self, PyObject* args) {
    injected_stream = NULL;
    Py_RETURN_NONE;
}
