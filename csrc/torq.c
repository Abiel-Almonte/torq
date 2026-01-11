#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <cuda_runtime.h>

#define _CUDA_CHECK(expr, message) { \
    cudaError_t _err = expr; \
    if (_err != cudaSuccess) { \
        PyErr_SetString(PyExc_RuntimeError, message); \
        return NULL; \
    } \
} \

static PyObject* device_sync(PyObject* self, PyObject *args){
    _CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize CUDA GPU");
    Py_RETURN_NONE;
}

static PyMethodDef _torq_methods [] = {
    {"device_sync", &device_sync, METH_NOARGS, "Synchronize CUDA GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _torq_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_torq",
    .m_doc = "Thin wrapper over the CUDA runtime for torq.",
    .m_methods = _torq_methods,

    .m_size = -1,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};

PyMODINIT_FUNC PyInit__torq(void){
    return PyModule_Create(&_torq_module);
}

/* TODO:
stream_create()
stream_destroy(stream)  
stream_sync(stream)

capture_begin(stream)
capture_end(stream, graph)

graph_instantiate(executor, graph)
graph_launch(executor, stream)
graph_destroy(executor, graph)
*/
