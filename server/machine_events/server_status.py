#!/usr/bin/env python3

import threading

from server.utils import (
	wait_time,
	EmptyLogger,
)

class ServerStatus(threading.Thread):
	def __init__(
		self,
		machines,
		refresh_interval=1,
		*,
		logger=None,
		**kwargs,
	):
		#super().__init__(**kwargs)
		threading.Thread.__init__(self, **kwargs)

		self.machines = machines
		self.refresh_interval = refresh_interval

		self.logger = logger if logger is not None else EmptyLogger()

		self._stop_signal = threading.Event()

	def export_summary(self, summary):
		self.logger.debug(summary)

	def run(self):
		while not self._stop_signal.is_set():
			for machine in self.machines:
				summary = machine.create_summary()
				self.export_summary(summary)
			self.logger.debug(f"Done exporting the summary of {len(self.machines)} machines, waiting for {self.refresh_interval}s.")
			wait_time(self.refresh_interval)

		self.logger.debug(f"ServerStatus is exiting.")

	def stop(self):
		self.logger.info(f"ServerStatus got signal to exit.")
		self._stop_signal.set()