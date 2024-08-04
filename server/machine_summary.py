from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .machine import Machine

from .machine_status import MachineStatus

# would be nicer to have a more broad definition
timestamp = int

@dataclass
class BaseMachineSummary(ABC):
	# Attributes (used by @dataclass to create __init__)

	machine: Machine
	benchmark: str
	status: MachineStatus = MachineStatus.UNKNOWN

@dataclass
class ActiveMachineSummary(BaseMachineSummary):
	benchmark_start: timestamp
	logs_per_sec: float
	iterations_per_sec: float
	sdc_count_total: int
	sdc_count_run: int

	status: MachineStatus = MachineStatus.ACTIVE


@dataclass
class RebootingMachineSummary(BaseMachineSummary):
	reboot_attempts: int
	last_active: timestamp
	last_reboot_attempt: timestamp
	max_reboot_attempts: int

	status: MachineStatus = MachineStatus.REBOOTING

@dataclass
class SleepingMachineSummary(BaseMachineSummary):
	last_active: timestamp
	last_reboot_attempt: timestamp
	next_reboot: timestamp

	status: MachineStatus = MachineStatus.SLEEPING