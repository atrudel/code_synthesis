class GameOfLife:
	def __init__(self, width = 0, height = 0):
		self._width = width
		self._height = height
		self._cycle = 0
		#the cell information are stored as a byte
		#bit 0 - cell owned by first player
		#bit 1 - cell owned by second player
		self._grid = bytearray(width * height)
		self._next_grid = bytearray(width * height)

	def set_point(self, x, y, value):
		if x < 0 or x >= self._width or y < 0 or x >= self._height:
			raise ValueError("coordinates out of bounds ({}, {})".format(x, y))
		self._grid[y * self._width + x] = value

	def set_grid(self, grid):
		if not isinstance(grid, bytearray):
			raise TypeError("set_grid requires a bytearray as argument")
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

def	step(grid, next_grid, width, height):
	cdef char *ptr = grid
	cdef char *next = next_grid
	cdef unsigned int x, y, w, h, n1, n2
	cdef char b

	w = width
	h = height
	for y in range(height):
		for x in range(width):
			n1 = 0
			n2 = 0
			if x > 0 and y > 0:			#top left
				b = ptr[(y - 1) * w + (x - 1)]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1
			if y > 0:					#top
				b = ptr[(y - 1) * w + x]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1
			if x < w and y > 0:			#top right
				b = ptr[(y - 1) * w + (x + 1)]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1
			if x > 0:					#left
				b = ptr[y * w + (x - 1)]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1
			if x < w - 1:				#right
				b = ptr[y * w + (x + 1)]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1
			if x > 0 and y < h - 1:		#bottom left
				b = ptr[(y + 1) * w + (x - 1)]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1
			if y < h - 1:				#bottom
				b = ptr[(y + 1) * w + x]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1
			if x < w - 1 and y < h - 1:	#bottom right
				b = ptr[(y + 1) * w + (x + 1)]
				if b & 1:
					n1 += 1
				elif b & 2:
					n2 += 1

			if ptr[y * w + x]:		#if alive
				if n1 + n2 < 2 or n1 + n2 > 3:
					next[y * w + x] = 0
				else:
					next[y * w + x] = ptr[y * w + x]
			else:					#if dead
				if n1 + n2 == 3:
					if n1 > n2:
						next[y * w + x] = 1
					else:
						next[y * w + x] = 2
				else:
					next[y * w + x] = ptr[y * w + x]
