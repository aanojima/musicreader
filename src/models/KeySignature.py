class KeySignature():

	def __init__(self, accidentals):
		self.accidentals = accidentals

	def get_accidentals(self):
		return self.accidentals

	def display(self):
		message = "KEY SIGNATURE - [ "
		for accidental in self.accidentals:
			message = message + "TYPE: " + str(accidental.get_type()) + ", PITCH: " + str(accidental.get_pitch()) + "; "
		message = message + "]"
		print message