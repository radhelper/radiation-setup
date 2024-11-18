import time
from .machine_status import MachineStatus
from .line_parser import (
	parse_it_line,
)
from .machine_summary import (
	ActiveMachineSummary,
	RebootingMachineSummary,
	SleepingMachineSummary,
)

from .utils import (
	safe_max,
)

timestamp = int

# TO-DO
# file with constants
MAX_CONSECUTIVE_HARD_REBOOTS = 6
SLEEP_TIME_AFTER_FAILED_REBOOTS = 30 * 60

class MachineEvents:
	def __init__(
		self,
		machine,
		logger,
	):
		self.machine = machine
		self.logger = logger

		self.status = MachineStatus.UNKNOWN

		# timestamps
		self.benchmark_start = None
		self.last_run_start = None
		self.last_run_end = None
		self.run_start = None

		# accumulated time
		self.benchmark_acc_time = 0
		self.run_acc_time = 0
		
		# Logs
		self.benchmark_sdcs = 0
		self.benchmark_logs = 0

		self.first_log_time = None
		self.last_log_time = None

		# Iterations

		self.benchmark_iterations = 0
		self.run_iterations = 0

		# SDCs
		self.run_sdcs = 0
		self.run_logs = 0
		
		self.first_sdc_time = None
		self.last_sdc_time = None

		# DUEs

		self.benchmark_dues = 0
		self.first_due_time = None
		self.last_due_time = None

		# reboots
		self.benchmark_soft_reboots = 0
		self.last_soft_reboot_time = None

		self.benchmark_hard_reboots = 0
		self.last_hard_reboot_time = None

	def start_benchmark(self, start_time=None):
		if start_time is None:
			start_time = time.time()

		if self.benchmark_start is not None:
			if self.logger is not None:
				self.logger.warning(f"Machine {self.machine_name} already has a benchmark start timestamp ({self.benchmark_start}). Did you mean to start a new run?")

		self.benchmark_start = start_time

	def start_run(self, start_time=None):
		if start_time is None:
			start_time = time.time()

		if self.run_start is not None:
			if self.logger is not None:
				self.logger.warning(f"Machine {self.machine_name} already has a run start timestamp ({self.benchmark_start}). Did you forget to end the run?")

		self.run_start = start_time
		self.run_logs = 0
		self.run_iterations = 0
		self.run_sdcs = 0
		self.run_acc_time = 0
		self.last_soft_reboot_time = None
		self.last_hard_reboot_time = None

	def end_run(self, end_time=None):
		if end_time is None:
			end_time = time.time()

		self.last_run_start = self.run_start
		self.last_run_end = end_time
		self.run_start = None
		self.benchmark_sdcs += self.run_sdcs
		self.benchmark_acc_time += self.run_acc_time
		self.benchmark_iterations += self.run_iterations

	def sdc(self, sdc_count=1, sdc_time=None):
		if sdc_time is None:
			sdc_time = time.time()

		self.run_sdcs += 1
		self.log(log_time=sdc_time)

	def iteration(self, info_entry, info_time=None):
		acc_time = info_entry['accumulated_time']
		iterations = info_entry['iterations']

		if info_time is None:
			info_time = time.time()

		self.run_acc_time = acc_time
		self.run_iterations = iterations
		self.log(log_time=info_time)

	def log(self, log_count=1, log_time=None):
		if log_time is None:
			log_time = time.time()

		if self.first_log_time is None:
			self.first_log_time = log_time

		self.last_log_time = log_time
		self.benchmark_logs += log_count
		self.run_logs += log_count

	def soft_reboot(self, reboot_count=1, reboot_time=None):
		if reboot_time is None:
			reboot_time = time.time()

		self.benchmark_soft_reboots += reboot_count
		self.last_soft_reboot_time = reboot_time

	def hard_reboot(self, reboot_count=1, reboot_time=None):
		if reboot_time is None:
			reboot_time = time.time()

		self.benchmark_hard_reboots += reboot_count
		self.last_hard_reboot_time = reboot_time

	def due(self, due_count=1, due_time=None):
		if due_time is None:
			due_time = time.time()

		self.benchmark_dues += 1

		if self.first_due_time is None:
			self.first_due_time = due_time

		self.last_due_time = due_time

		self.end_run(end_time=due_time)

	def handle_event(self, event_type, event_message=None):
		try:
			self._handle_event(event_type, event_message)
		except ValueError as e:
			if self.logger is not None:
				self.logger.info(f"MachineEvents for {self.machine_name} could not parse event of type {event_type}.")

	def _handle_event(self, event_type, event_message=None):
		if event_type == '#IT':
			line_values = parse_it_line(event_message)
			self.iteration(line_values)
		elif event_type == '#HEADER':
			self.start_run()
		elif event_type == '#END':
			self.end_run()
		elif event_type == '#INF':
			self.log()
		elif event_type == '#ERR':
			self.log()
		elif event_type == "#SDC":
			self.sdc()
		elif event_type == "#ABORT":
			self.due()
		elif event_type == "#LOGFILE":
			pass
		else:
			raise ValueError(f"Invalid event type: {event_type}")

	@property
	def consecutive_soft_reboots(self):
		return self.machine.soft_app_reboot_count

	@property
	def consecutive_hard_reboots(self):
		return self.machine.hard_reboot_count

	@property
	def machine_name(self):
		return self.machine.machine_name

	@property
	def machine_info(self):
		return str(self.machine)

	@property
	def benchmark(self):
		return self.machine_name

	def create_summary(self):
		status = MachineStatus.UNKNOWN

		if self.run_start is not None:
			status = MachineStatus.ACTIVE
		elif self.consecutive_hard_reboots < MAX_CONSECUTIVE_HARD_REBOOTS and self.consecutive_soft_reboots > 0:
			status = MachineStatus.REBOOTING
		elif self.consecutive_hard_reboots == MAX_CONSECUTIVE_HARD_REBOOTS:
			status = MachineStatus.SLEEPING

		if status == MachineStatus.ACTIVE:
			benchmark_duration = time.time() - self.benchmark_start
			if benchmark_duration > 0:
				logs_per_sec = self.benchmark_logs / benchmark_duration
			else:
				logs_per_sec = 0

			run_duration = time.time() - self.run_start

			if run_duration > 0:
				iterations_per_sec = self.run_iterations / run_duration
			else:
				iterations_per_sec = 0

			summary = ActiveMachineSummary(
				self.machine,
				self.benchmark,
				self.benchmark_start,
				logs_per_sec,
				iterations_per_sec,
				self.benchmark_sdcs,
				self.run_sdcs,
			)
		elif status == MachineStatus.REBOOTING:
			last_reboot_attempt = safe_max(self.last_hard_reboot_time, self.last_soft_reboot_time)
			reboot_attempts = safe_max(self.consecutive_soft_reboots, self.consecutive_hard_reboots)
			summary = RebootingMachineSummary(
				self.machine,
				self.benchmark,
				reboot_attempts,
				self.last_run_end, #last_active
				last_reboot_attempt,
				MAX_CONSECUTIVE_HARD_REBOOTS,
			)
		elif status == MachineStatus.SLEEPING:
			last_reboot_attempt = safe_max(self.last_hard_reboot_time, self.last_soft_reboot_time)
			next_reboot = last_reboot_attempt + SLEEP_TIME_AFTER_FAILED_REBOOTS
			summary = SleepingMachineSummary(
				self.machine,
				self.benchmark,
				self.last_run_end, #last_active
				last_reboot_attempt,
				next_reboot,
			)
		else:
			summary = None

		return summary