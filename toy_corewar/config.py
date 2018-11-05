import torch
import json

class _store():
	cfg = None

def load(filename):
	with open(filename, 'r') as f:
		data = json.load(f)
	_store.cfg = Cfg(data)
	return _store.cfg

def store(filename):
	data = _store.cfg.todict()
	with open(filename, 'w') as f:
		json.dump(data, f, indent = 4)

def get_cfg():
	return _store.cfg

def populate(obj, data):
	if type(data) is dict:
		for key, value in data.items():
			if type(value) is dict:
				target = Container()
				setattr(obj, key, target)
				populate(target, value)
			else:
				setattr(obj, key, value)

def retrieve(obj):
	for k, v in obj.items():
		if type(v) is Container:
			obj[k] = retrieve(vars(v))
	return obj

class Container(object):
	def todict(self):
		return retrieve(vars(self))

class Cfg(object):
	def __init__(self, data):
		populate(self, data)

	def todict(self):
		return retrieve(vars(self))

# Hardware usage
DEVICE = 'cpu'
