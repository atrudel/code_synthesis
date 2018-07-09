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
class GameOfLife:
	def __init__(self, width = 0, height = 0):
		self._width = width
		self._height = height
		self._cycle = 0
		#the cell information are stored as a byte
		#bit 0 - cell owned by first player
		#bit 1 - cell owned by second player
		self._grid = np.zeros(shape=(self._height, self._width), dtype=np.byte)
		self._next_grid = np.zeros(shape=(self._height, self._width), dtype=np.byte)
		self._points = set()

	def set_point(self, x, y, value):
		if x < 0 or x >= self._width or y < 0 or y >= self._height:
			raise ValueError("coordinates out of bounds ({}, {})".format(x, y))
		self._grid[y][x] = value
		self._points.add((x, y))

	def set_grid(self, grid):
		#if not isinstance(grid, np.array):
		#	raise TypeError("set_grid requires a numpy array as argument")
		if grid.shape != self._grid.shape:
			raise ValueError("grid size doesn't match the game size ({} and {}".format(grid.shape, self._grid.shape))
		self._grid = grid
		self.set_points()

	def set_points(self):
		self._points.clear()
		for y in range(self._grid.shape[0]):
			for x in range(self._grid.shape[1]):
				if self._grid[y][x] != 0:
					self._points.add((x, y))

	def grid(self):
		return self._grid

	def run_steps(self, steps):
		self._cycle += steps
		for i in range(steps):
			step(self._grid, self._next_grid, self._width, self._height, self._points)
			self._grid = self._next_grid
			self._next_grid = np.zeros(shape=(self._height, self._width), dtype=np.byte)
		#print (numba.typeof(self._points))

	def size(self):
		return (self._width, self._height)

@jit(cache=True)
def	step(grid, next_grid, w, h, points):
	born = set()
	dead = set()
	for bx, by in points:
		xmin = bx - 1 if bx > 0 else bx
		xmax = bx + 1 if bx < w - 1 else bx
		ymin = by - 1 if by > 0 else by
		ymax = by + 1 if by < h - 1 else by
		for y in range(ymin, ymax + 1):
			for x in range(xmin, xmax + 1):
				n1 = 0
				n2 = 0

				if x > 0 and y > 0:			#top left
					b = grid[y - 1][x - 1]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1
				if y > 0:					#top
					b = grid[y - 1][x]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1
				if x < w - 1 and y > 0:		#top right
					b = grid[y - 1][x + 1]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1
				if x > 0:					#left
					b = grid[y][x - 1]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1
				if x < w - 1:				#right
					b = grid[y][x + 1]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1
				if x > 0 and y < h - 1:		#bottom left
					b = grid[y + 1][x - 1]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1
				if y < h - 1:				#bottom
					b = grid[y + 1][x]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1
				if x < (w - 1) and y < (h - 1):	#bottom right
					b = grid[y + 1][x + 1]
					if b == 1:
						n1 += 1
					elif b == 2:
						n2 += 1

				if grid[y][x]:		#if alive
					if n1 + n2 < 2 or n1 + n2 > 3:
						next_grid[y][x] = 0
						dead.add((x, y))
					else:
						next_grid[y][x] = grid[y][x]
				else:					#if dead
					if n1 + n2 == 3:
						if n1 > n2:
							next_grid[y][x] = 1
						else:
							next_grid[y][x] = 2
						born.add((x, y))
					else:
						next_grid[y][x] = grid[y][x]
	points.difference_update(dead)
	points.update(born)
