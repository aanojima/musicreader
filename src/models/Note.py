class Note():

	def __init__(self):
		self.pitch = None;
		self.type = None;
		self.modifiers = None;
		pass

class Pitch:
	A,B,C,D,E,F,G = range(7)

class Type:
	Whole, Half, Quarter, Eighth, Rest = range(5)