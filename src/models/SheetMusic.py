from models.Accidental import *
from models.Clef import *
from models.Note import *
from models.TimeSignature import *
from models.KeySignature import *

from modules.staves import *
from modules.common import *

class SheetMusic():

	def __init__(self, sheet, symbol_array):
		self.sheet = sheet
		self.notes = []
		self.current_accidental = None
		self.time_signature_count = None
		self.time_signature_type = None
		self.KEY_SIGNATURE = []
		for symbol in symbol_array:
			dtype = symbol['type']
			label = symbol['label']
			box = symbol['box']
			data = symbol['data']
			if dtype == Symbol.CLEF:
				self.set_clef(label)
			elif dtype == Symbol.KEY_SIGNATURE:
				# TODO:
				symbol['data'] = []
				accidentals = []
				for i in range(len(label)):
					l = label[i]
					b = box[i]
					if is_accidental(l):
						line_index = sheet.get_line(Symbol.ACCIDENTAL, b, l)
						accidental = Accidental(l, line_index)
						accidentals.append(accidental)
						symbol['data'].append(line_index)
				self.set_key_signature(accidentals)
			elif dtype == Symbol.TIME_SIGNATURE:
				self.set_time_signature(label, data)
			elif dtype == Symbol.NOTE:
				line_index = sheet.get_line(dtype, box, label)
				symbol['data'] = line_index
				if is_note(label):
					self.add_note(label, line_index)
				elif is_accidental(label):
					self.apply_accidental(label, line_index)

	def set_clef(self, clef_type):
		self.clef = Clef(clef_type)

	# Requires a list of accidentals
	def set_key_signature(self, accidentals):
		self.key_signature = KeySignature(accidentals)

	def set_time_signature(self, ts_type, data):
		if ts_type == TimeSignatureLabel.COUNT:
			self.time_signature_count = data
		elif ts_type == TimeSignatureLabel.TYPE:
			self.time_signature_type = data
		if self.time_signature_count is not None and self.time_signature_type is not None:
			self.time_signature = TimeSignature(self.time_signature_count, self.time_signature_type)

	def apply_accidental(self, a_type, line_index):
		pitch = line_index # TODO: Find mapping
		self.current_accidental = Accidental(a_type, pitch)

	def reset_accidental(self):
		self.current_accidental = None

	# ADD Note (with current accidental) and then reset current accidental to None
	def add_note(self, note_type, line_index):
		pitch = line_index # TODO: Find pitch from line_index
		note = Note(note_type, pitch) # TODO
		if self.current_accidental is not None:
			note.apply_accidental(self.current_accidental)
		self.notes.append(note)
		self.reset_accidental()

	def display(self):
		self.clef.display()
		self.key_signature.display()
		self.time_signature.display()
		for note in self.notes:
			note.display()