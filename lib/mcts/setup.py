from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the absolute path to the .pyx file
pyx_file = os.path.join(os.path.dirname(__file__), 'mcts_cython.pyx')

extensions = [
    Extension(
        "mcts_cython",
        sources=[pyx_file],  # Use the full path
        extra_compile_args=["-O3", "-ffast-math"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="MCTS Cython",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)