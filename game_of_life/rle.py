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
		for line in data:
			arr = line_to_byte(line, self.width)
			self.data.extend(arr)

def line_to_byte(line, width):
	b = bytearray()
	last = 0
	i = 0
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
			b.extend(bytearray([data] * count))
			last = i + 1
		i += 1

	b.extend(bytearray(width - len(b)))
	if i > last:
		count = int(line[last:i]) - 1
		b.extend(bytearray(width * count))
	return b
