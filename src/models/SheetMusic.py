from models.Accidental import *
from models.Clef import *
from models.Note import *
from models.TimeSignature import *

from modules.note import *

class SheetMusic():

	self.clef = None
	self.time_signature = None
	self.time_signature_count = None
	self.time_signature_type = None
	self.current_accidental = None
	self.notes = []

	def __init__(self, symbol_array):
		for i in symbol_array:
			symbol = symbol_array[i]
			if symbol['type'] == CLEF: # TODO: Create Enums/Constants for these
				# Ignore Getting Line ==> straight to data model construction
				clef = symbol['type']
				self.set_clef(clef)
			# TODO: What about Key Signature?
			elif symbol['type'] == TIME_SIGNATURE:
				# TODO: CLASSIFIER should return COUNT or TYPE
				ts_count = symbol['data']
				self.set_time_sinature_count(ts_count)
			elif symbol['type'] == TIME_SIGNATURE_TYPE:
				ts_type = symbol['data']
				self.set_time_signature_type(ts_type)
			elif symbol['type'] in [NATURAL, SHARP, FLAT]:
				line_index = staves.get_line(symbol['type'], symbol['box'])
				self.apply_accidental(symbol['type'], line_index)
			elif symbol['type'] in [EIGHTH, QUARTER, HALF, WHOLE]
				line_index = staves.get_line(symbol['type'], symbol['box'])
				self.add_note(symbol['type'], line_index)

	def set_clef(self, clef_type):
		self.clef = Clef(clef_type)

	# Requires a list of accidentals
	def set_key_signature(self, accidentals):
		pass

	def set_time_signature_count(self, count):
		self.time_signature_count = count
		if self.time_signature_count is not None and self.time_signature_type is not None:
			self.time_signature = TimeSignature(self.time_signature_count, self.time_signature_type)

	def set_time_signature_type(self, ts_type):
		self.time_signature_type = ts_type
		if self.time_signature_count is not None and self.time_signature_type is not None:
			self.time_signature = TimeSignature(self.time_signature_count, self.time_signature_type)

	def apply_accidental(self, a_type, line_index):
		pitch = line_index # TODO: Find mapping
		current_accidental = Accidental(a_type, pitch)

	# ADD Note (with current accidental) and then reset current accidental to None
	def add_note(self, note_type, line_index):
		pitch = line_index # TODO: Find pitch from line_index
		note = Note(note_type, pitch) # TODO
		# TODO: Apply any accidentals
		self.notes.push(note)