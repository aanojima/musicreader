from models.Accidental import *
from models.Clef import *
from models.Note import *
from models.TimeSignature import *

from modules.common import *

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
			dtype = symbol['type']
			label = symbol['label']
			box = symbol['box']
			data = symbol['data']
			if dtype == Symbol.CLEF: # TODO: Create Enums/Constants for these
				# Ignore Getting Line ==> straight to data model construction
				self.set_clef(label)
			# TODO: What about Key Signature?
			elif dtype == Symbol.TIME_SIGNATURE:
				# TODO: CLASSIFIER should return COUNT or TYPE
				self.set_time_sinature(label, data)
			elif dtype == Symbol.Accidental:
				line_index = staves.get_line(dtype, symbol['box'])
				self.apply_accidental(label, line_index)
			elif dtype == Sybmol.Note:
				line_index = staves.get_line(dtype, box)
				self.add_note(label, line_index)

	def set_clef(self, clef_type):
		self.clef = Clef(clef_type)

	# Requires a list of accidentals
	def set_key_signature(self, accidentals):
		pass

	def set_time_signature(self, ts_type, data):
		if ts_type == TimeSignatureLabel.COUNT:
			self.time_signature_count = data
		elif ts_type == TimeSignatureLabel.TYPE:
			self.time_signature_type = data
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
		if current_accidental is not None:
			pass # TODO: Apply accidental to note
		self.notes.push(note)
		current_accidental = None