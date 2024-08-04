import enum

class MachineStatus(enum.Enum):
	'''
		List of possible status for each machine.
		The status dictates what attributes will be in the summary.
	'''
	ACTIVE = enum.auto()
	REBOOTING = enum.auto()
	SLEEPING = enum.auto()
	UNKNOWN = enum.auto()