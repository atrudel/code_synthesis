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
		self.data = bytearray()
		offset = 0
		for line in data:
			arr = line_to_byte(line)
			arr.extend(bytearray(self.width - len(arr)))
			self.data.extend(arr)


def line_to_byte(line):
	b = bytearray()
	last = 0
	for i in range(len(line)):
		if line[i] != "b" and line[i] != "o":
			continue

		if line[i] == "b":
			data = 0
		else:
			data = 1

		count = 1
		if i > last:
			count = int(line[last:i])
		b.extend(bytearray([data] * count))
		last = i + 1
	return b
