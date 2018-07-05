from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
	Extension("gol", ["src/gol.pyx"])
]

setup(name='Game of life',
      ext_modules=cythonize(extensions))

