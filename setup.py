from setuptools import setup, Extension


ctorq_extension = Extension(
    name="torq.cuda._torq",
    sources=["csrc/torq.c"],
    include_dirs=[
        "./csrc",
        "/usr/include/cuda/",
        "/usr/include/python3.12/"
        #cuda_runtime.h is already in /usr/include/
    ],
    library_dirs=["/usr/local/cuda/lib64"],
)

setup(
    name="torq",
    packages=["torq", "torq.cuda"],
    package_data={
        "torq": ["*.pyi", "py.typed"],
        "torq.cuda": ["*.pyi", "py.typed"]
    },
    ext_modules=[ctorq_extension]
)