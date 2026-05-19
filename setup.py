from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="torch_persistent_homology_cpu",
    ext_modules=[
        CppExtension(
            name="torch_persistent_homology_cpu",
            sources=["torch_persistent_homology/persistent_homology_cpu.cpp"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
