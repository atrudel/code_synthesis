import numpy as np
from numpy import byte
import re

class Pattern:
	def __init__(self, filename):
		f = open(filename)
		line = f.readline()
		while line.startswith("#"):
			line = f.readline()
		match = re.search("x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)", line)
		if match and len(match.groups()) == 2:
			self.width = int(match.group(1))
			self.height = int(match.group(2))
		else:
			raise Exception("cannot parse pattern size")
		data = "".join(f.readlines()).replace("\n", "")
		data = re.findall("([^$!]*.)", data)

		def make_line(string):
			result = np.zeros((self.width, ), dtype=byte)
			match = re.findall("\s*(\d*\s*[bo$!])\s*", string)
			pos = 0
			for tag in match:
				size = 1
				amount, elem = re.match("(\d*)\s*([ob$!])", tag).groups()
				if len(amount) > 0:
					size = int(amount)
					if size == 0:
						raise Exception("invalid repetition count")

				if elem == "o":
					result[pos : pos + size] = np.ones((size, ), dtype=byte)
					pos += size
				elif elem == "b":
					pos += size
				elif elem == "$":
					result = np.vstack((result, \
						np.zeros((size - 1, result.shape[0]), dtype=byte)))
					break
				elif elem == "!":
					break
			return result

		self.data = None
		for line in data:
			result = make_line(line)
			if self.data is None:
				self.data = result
			else:
				self.data = np.vstack((self.data, result))

def pad_pattern(pattern, shape):
	if len(pattern.shape) != len(shape) or len(pattern.shape) != 2:
		raise Exception("wrong pattern or shape")

	if (pattern.shape[0] > shape[0] or pattern.shape[1] > shape[1]):
		raise Exception("pattern {} is bigger than target shape {}".format(pattern.shape, shape))

	hdiff = shape[1] - pattern.shape[1]
	vdiff = shape[0] - pattern.shape[0]
	hpad = (hdiff) // 2
	vpad = (vdiff) // 2
	return np.pad(pattern, ((vpad, vpad + 1 if vdiff % 2 else vpad),\
		(hpad, hpad + 1 if hdiff % 2 else hpad)),\
		'constant', constant_values=0)
