from numba import jit
from numba import jitclass
import numba
import numpy as np

types = [
	('_width', numba.int32),
	('_height', numba.int32),
	('_cycle', numba.int32),
	('_grid', numba.byte[:]),
	('_next_grid', numba.byte[:])
	]

#@jitclass(types)
class GameOfLife(object):
	def __init__(self, width = 0, height = 0):
		self._width = width
		self._height = height
		self._cycle = 0
		#the cell information are stored as a byte
		#bit 0 - cell owned by first player
		#bit 1 - cell owned by second player
		self._grid = np.zeros(shape=(self._width * self._height), dtype=np.byte)
		self._next_grid = np.zeros(shape=(self._width * self._height), dtype=np.byte)

	def set_point(self, x, y, value):
		if x < 0 or x >= self._width or y < 0 or y >= self._height:
			raise ValueError("coordinates out of bounds ({}, {})".format(x, y))
		self._grid[y * self._width + x] = value

	def set_grid(self, grid):
		#if not isinstance(grid, np.array):
		#	raise TypeError("set_grid requires a numpy array as argument")
		if len(grid) != self._width * self._height:
			raise ValueError("grid size doesn't match the game size")
		self._grid = grid

	def grid(self):
		return self._grid

	def run_steps(self, steps):
		self._cycle += steps
		for i in range(steps):
			step(self._grid, self._next_grid, self._width, self._height)
			self._grid, self._next_grid = self._next_grid, self._grid

	def size(self):
		return (self._width, self._height)

@jit(cache=True)
def	step(grid, next_grid, width, height):
	w = width
	h = height

	for y in range(height):
		for x in range(width):
			n1 = 0
			n2 = 0
			if x > 0 and y > 0:			#top left
				b = grid[(y - 1) * w + (x - 1)]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1
			if y > 0:					#top
				b = grid[(y - 1) * w + x]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1
			if x < w - 1 and y > 0:		#top right
				b = grid[(y - 1) * w + (x + 1)]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1
			if x > 0:					#left
				b = grid[y * w + (x - 1)]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1
			if x < w - 1:				#right
				b = grid[y * w + (x + 1)]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1
			if x > 0 and y < h - 1:		#bottom left
				b = grid[(y + 1) * w + (x - 1)]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1
			if y < h - 1:				#bottom
				b = grid[(y + 1) * w + x]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1
			if x < (w - 1) and y < (h - 1):	#bottom right
				b = grid[(y + 1) * w + (x + 1)]
				if b == 1:
					n1 += 1
				elif b == 2:
					n2 += 1

			if grid[y * w + x]:		#if alive
				if n1 + n2 < 2 or n1 + n2 > 3:
					next_grid[y * w + x] = 0
				else:
					next_grid[y * w + x] = grid[y * w + x]
			else:					#if dead
				if n1 + n2 == 3:
					if n1 > n2:
						next_grid[y * w + x] = 1
					else:
						next_grid[y * w + x] = 2
				else:
					next_grid[y * w + x] = grid[y * w + x]
