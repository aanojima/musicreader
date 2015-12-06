class TimeSignature():

	TIME_SIGNATURE_COUNT = 0
	TIME_SIGNATURE_TYPE = 1

	self.ts_count = None
	self.ts_type = None

	def __init__(self, ts_count, ts_type):
		self.ts_count = ts_count
		self.ts_type = ts_type