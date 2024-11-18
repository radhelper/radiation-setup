import time

# Standardized function to wait for some time
def wait_time(duration):
	# time.sleep() is not ideal
	# TO-DO: replace it with something better
	time.sleep(duration)

class EmptyLogger:
	def debug(self, *args, **kwargs):
		pass

	def info(self, *args, **kwargs):
		pass

	def perf(self, *args, **kwargs):
		pass

	def warning(self, *args, **kwargs):
		pass

	def error(self, *args, **kwargs):
		pass

	def critical(self, *args, **kwargs):
		pass

def safe_max(a, b):
	if a is None:
		return b
	elif b is None:
		return a
	else:
		return max(a,b)