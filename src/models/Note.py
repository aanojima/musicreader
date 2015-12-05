class Note():

	def __init__(self):
		self.pitch = None;
		self.type = None;
		self.modifiers = None;
		pass


	def set_pitch(self, type):
		pass


	def set_type(self, type):
		pass


	def set_modifiers(self, type):
		pass

class Pitch:
	A,B,C,D,E,F,G = range(7)

class Type:
	Whole, Half, Quarter, Eighth, Rest = range(5)