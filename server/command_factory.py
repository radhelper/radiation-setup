import collections
import json
import logging
import time
import typing

from logger_formatter import logging_setup

_ONE_HOUR_WINDOW = 3600


class CommandFactory:
    def __init__(self, json_files_list: list, logger_name: str, command_window: int = _ONE_HOUR_WINDOW):
        self.__current_command_kill = None
        self.__command_window = command_window
        self.__json_data_list = list()
        self.__logger = logging.getLogger(f"{logger_name}.{__name__}")
        for json_file in json_files_list:
            try:
                with open(json_file) as fp:
                    machine_dict = json.load(fp)
                    # The json files contains a list of dicts
                    self.__json_data_list.extend(machine_dict)
            except FileNotFoundError:
                self.__logger.exception(f"Incorrect path for {json_file}, file not found")
                raise

        # Transform __json_data_list into a FIFO to manage the codes testing
        self.__cmd_queue = collections.deque()
        self.__check_and_refill_the_queue()
        self.__current_command = self.__cmd_queue.pop()
        self.__current_command["start_timestamp"] = time.time()

    def __check_and_refill_the_queue(self):
        """ Fill or re-fill the command queue """
        if not self.__cmd_queue:
            self.__logger.debug("Re-filling the queue of commands")
            self.__cmd_queue = collections.deque(self.__json_data_list)

    @property
    def current_cmd_kill(self) -> bytes:
        return self.__current_command_kill

    @property
    def is_command_window_timeout(self):
        """ Only checks if the self.__current_command is outside execute window
        :return:
        """
        now = time.time()
        time_diff = now - self.__current_command["start_timestamp"]
        return time_diff > self.__command_window

    def get_commands_and_test_info(self, encode: str = 'ascii') -> typing.Tuple[str, str, str]:
        """ Based on a Factory pattern we can build the string taking into consideration how much a cmd already
        executed. For example, if we have 10 configurations on the __json_data_list, then the get_cmd will
        select the one that is currently executing and did not complete __command_window time.
        :param encode: encode type, default ascii
        :return: cmd_exec and cmd_kill encoded strings
        """
        self.__check_and_refill_the_queue()

        # verify the timestamp first
        if self.is_command_window_timeout:
            self.__current_command = self.__cmd_queue.pop()
            self.__current_command["start_timestamp"] = time.time()
        cmd_exec = self.__current_command["exec"].encode(encoding=encode)
        cmd_kill = self.__current_command["killcmd"].encode(encoding=encode)
        self.__current_command_kill = cmd_kill
        code_name = self.__current_command["codename"]
        code_header = self.__current_command["header"]
        return cmd_exec, code_name, code_header


if __name__ == '__main__':
    def debug():
        # FOR DEBUG ONLY
        logger = logging_setup(logger_name="COMMAND_FACTORY", log_file="unit_test_log_CommandFactory.log")
        logger.debug("DEBUGGING THE COMMAND FACTORY")
        logger.debug("CREATING THE MACHINE")
        command_factory = CommandFactory(json_files_list=["../machines_cfgs/cuda_micro.json"],
                                         logger_name="COMMAND_FACTORY",
                                         command_window=5)

        logger.debug("Executing command factory")
        first = command_factory.get_commands_and_test_info()[0]
        for it in range(20):
            time.sleep(2)
            sec = command_factory.get_commands_and_test_info()[0]
            if first == sec:
                logger.debug(f"-------- IT {it} EQUAL AGAIN ----------------")
        time.sleep(1)
        logger.debug(str(command_factory.is_command_window_timeout))


    debug()
