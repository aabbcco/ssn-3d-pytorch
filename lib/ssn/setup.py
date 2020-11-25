from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="pair_wise_distance",
    ext_modules=[cpp_extension.CppExtension(
        'pair_wise_distance', ['pair_wise_distance_cuda_source.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
