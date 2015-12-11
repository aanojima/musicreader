class TimeSignature():

	def __init__(self, ts_count, ts_type):
		self.ts_count = ts_count
		self.ts_type = ts_type

	def display(self):
		print "TIME SIGNATURE - COUNT: " + str(self.ts_count) + ", TYPE: " + str(self.ts_type)