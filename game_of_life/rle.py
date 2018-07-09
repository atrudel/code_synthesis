import numpy as np

class Pattern:
	def __init__(self, filename):
		f = open(filename)
		line = f.readline()
		while line.startswith("#"):
			line = f.readline()
		coords = line.split(",")
		self.width = int(coords[0][4:])
		self.height = int(coords[1][5:])
		data = "".join(f.readlines()).replace("\n", "")
		data = data[:data.find("!")]
		data = data.split("$")
		self.data = np.zeros(0, dtype=np.byte)
		for line in data:
			arr = line_to_byte(line, self.width)
			if len(self.data) == 0:
				self.data = arr
			else:
				self.data = np.vstack((self.data, arr))

def line_to_byte(line, width):
	b = np.zeros((width,), dtype = np.byte)
	last = 0
	i = 0
	pos = 0
	if len(line) != 0:
		for i in range(len(line)):
			if line[i].isdigit():
				continue

			elif line[i] == "b":
				data = 0
			else:
				data = 1

			count = 1
			if i > last:
				count = int(line[last:i])
			for k in range(count):
				b[pos + k] = data
			pos += count
			last = i + 1
		i += 1

	if i > last:
		count = int(line[last:i]) - 1
		b = np.vstack((b, np.zeros((count, width), dtype = np.byte)))
	return b
