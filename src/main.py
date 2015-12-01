from models.Accidental import Accidental
from models.Clef import Clef
from models.Note import Note
from models.TimeSignature import TimeSignature
from modules.staves import *

def main():
	print "Hello World"
	a = Accidental()
	c = Clef()
	n = Note()
	ts = TimeSignature()
	find_stave_lines()

if __name__ == '__main__':
	main()