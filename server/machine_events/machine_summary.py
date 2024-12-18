import typing

from abc import ABC

from dataclasses import (
	dataclass,
	KW_ONLY,
)

from server.machine_events.machine_status import (
	MachineStatus,
)

# would be nicer to have a more broad definition
timestamp = int

@dataclass
class BaseMachineSummary(ABC):
	# Attributes (used by @dataclass to create __init__)

	machine: typing.Any
	benchmark: str
	_: KW_ONLY
	status: MachineStatus

@dataclass
class UnknownMachineSummary(BaseMachineSummary):
	machine: typing.Any
	benchmark: str
	_: KW_ONLY
	status: MachineStatus = MachineStatus.UNKNOWN

	def __str__(self):
		return f"The status of machine {self.machine.machine_name} is currently unknown, but it is expected to run benchmark {self.benchmark}.\n" + \
		"An unknown status might mean it is just starting, or something else."

	def __repr__(self):
		return str(self)

@dataclass
class BootingMachineSummary(BaseMachineSummary):
	machine: typing.Any
	benchmark: str
	_: KW_ONLY
	status: MachineStatus = MachineStatus.BOOTING

	def __str__(self):
		return f"Machine {self.machine.machine_name} is currently booting, and it is expected to run benchmark {self.benchmark}.\n" + \
		"."

	def __repr__(self):
		return str(self)

@dataclass
class ActiveMachineSummary(BaseMachineSummary):
	benchmark_start: timestamp
	logs_per_sec: float
	iterations_per_sec: float
	sdc_count_total: int
	sdc_count_run: int
	last_log_time: int
	_: KW_ONLY
	status: MachineStatus = MachineStatus.ACTIVE

	def __str__(self):
		return f"Machine {self.machine.machine_name} is running benchmark {self.benchmark}.\n" + \
		f"It is currently active with {self.logs_per_sec} logs/s, having started at {self.benchmark_start}.\n" + \
		f"On average, it is running {self.iterations_per_sec} iters/s, with a total of {self.sdc_count_total} SDCs ({self.sdc_count_run} in this run).\n" + \
		f"The last log was received at {self.last_log_time}"

	def __repr__(self):
		return str(self)


@dataclass
class RebootingMachineSummary(BaseMachineSummary):
	reboot_attempts: int
	last_active: timestamp
	last_reboot_attempt: timestamp
	max_reboot_attempts: int
	_: KW_ONLY
	status: MachineStatus = MachineStatus.REBOOTING

	def __str__(self):
		return f"Machine {self.machine.machine_name} is running benchmark {self.benchmark}.\n" + \
		f"It is currently rebooting, with the last reboot attempt having happened at {self.last_reboot_attempt}.\n" + \
		f"This is the reboot attempt #{self.reboot_attempts} (max. {self.max_reboot_attempts}), and was last active at {self.last_active}."

	def __repr__(self):
		return str(self)

@dataclass
class SleepingMachineSummary(BaseMachineSummary):
	last_active: timestamp
	last_reboot_attempt: timestamp
	next_reboot: timestamp
	_: KW_ONLY
	status: MachineStatus = MachineStatus.SLEEPING

	def __str__(self):
		return f"Machine {self.machine.machine_name} is running benchmark {self.benchmark}.\n" + \
		f"It is currently sleeping, with the last reboot attempt having happened at {self.last_reboot_attempt}.\n" + \
		f"The next reboot attempt will be at {self.next_reboot}, and it was last active at {self.last_active}."

	def __repr__(self):
		return str(self)