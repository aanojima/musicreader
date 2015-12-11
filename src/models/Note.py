class Note():

	def __init__(self, note_type, pitch):
		self.pitch = pitch
		self.type = note_type;
		self.modifiers = None
		self.accidental = None

	def set_pitch(self, type):
		pass


	def set_type(self, type):
		pass


	def set_modifiers(self, type):
		pass

	def apply_accidental(self, accidental):
		if accidental.get_pitch() == self.pitch:
			self.accidental = accidental
		else:
			print "Unmatched accidental"

	def display(self):
		message = "NOTE - LENGTH: " + str(self.type) + ", PITCH: " + str(self.pitch)
		if self.accidental is not None:
			 message = message + ", ACCIDENTAL: " + str(self.accidental.get_type())
		print message