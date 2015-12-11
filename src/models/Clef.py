from modules.common import *

class Clef():

	def __init__(self, clef):
		self.clef_type = clef
		if self.clef_type == ClefLabel.BASS:
			self.min = 43
			self.max = 57
		elif self.clef_type == ClefLabel.TREBLE:
			self.min = 64
			self.max = 77


	def get_type(self):
		return self.clef_type

	def display(self):
		if self.clef_type == ClefLabel.BASS:
			print "CLEF - BASS"
		elif self.clef_type == ClefLabel.TREBLE:
			print "CLEF - TREBLE"
