class Accidental():

	NATURAL = 0
	SHARP = 1
	FLAT = 2

	self.a_type = None
	self.pitch = None

	def __init__(self, a_type, pitch):
		self.a_type = a_type
		self.pitch = pitch