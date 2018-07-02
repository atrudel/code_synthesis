from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
	Extension("gol", ["src/game_of_life.pyx"])
]

setup(name='Hello world app',
      ext_modules=cythonize(extensions))

