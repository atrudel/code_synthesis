import torch
import json

class _store():
	cfg = None

def load(filename):
	with open(filename) as f:
		data = json.load(f)
	_store.cfg = Cfg(data)
	return _store.cfg

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

class Container(object):
	pass

class Cfg(object):
	def __init__(self, data):
		populate(self, data)

# Hardware usage
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
