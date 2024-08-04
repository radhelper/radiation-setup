import parse

def deduce_type(value):
	# check if int
	try:
		v = int(value)
		return v
	except ValueError:
		pass

	#check if float
	try:
		v = float(value)
		return v
	except ValueError:
		pass

	# check if any common bool string representations
	v = value.lower()
	if v == 'true' or v == 'yes' or v == 'enabled':
		return True

	if v == 'false' or v == 'no' or v == 'disabled':
		return False

	# we do not know what it is, return the string
	return value

def parse_it_line(text):
	assert text.startswith('#IT')
	line_format = "{iter:d} KerTime:{ker_time:f} AccTime:{acc_time:f}"
	vals = parse.parse(line_format, text)

	return {
		'iteration': vals['iter'],
		'kernel_time': vals['ker_time'],
		'accumulated_time': vals['acc_time'],
	}