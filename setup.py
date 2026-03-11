from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='RcMoE_prefetcher',
    ext_modules=[CUDAExtension('RcMoE_prefetcher', ['prefetcher.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
