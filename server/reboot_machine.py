"""
Reboot machine functions. This is conceptually different from
the radiation_benchmarks setup. Here we use only private functions, and the only
public function is reboot_machine, which will reboot based on the
parameters.
"""

import json
import logging
import os
import time
import typing

import requests

from error_codes import ErrorCodes

# Switches status, only used in this module
__ON = "ON"
__OFF = "OFF"


def _execute_command(cmd: str) -> ErrorCodes:
    """ Simple function to execute a shell command
    :param cmd: command string
    :return: ErrorCodes enum
    """
    tmp_file = "/tmp/server_error_execute_command"
    result = os.system(f"{cmd} 2>{tmp_file}")
    with open(tmp_file) as err:
        if len(err.readlines()) != 0 or result != 0:
            return ErrorCodes.GENERAL_ERROR
    return ErrorCodes.SUCCESS


def _lindy_switch(status: str, switch_port: int, switch_ip: str, logger: logging.Logger) -> ErrorCodes:
    """ Lindy switch reboot rules
    :param status: ON or OFF
    :param switch_port: port to reboot
    :param switch_ip: ip address for the switch
    :param logger: logging.Logger obj
    :return: ErrorCodes enum
    """
    to_change = "000000000000000000000000"
    led = f"{to_change[:(switch_port - 1)]}1{to_change[switch_port:]}"

    if status == __ON:
        # TODO: Check if lindy switch accepts https protocol
        url = f'http://{switch_ip}/ons.cgi?led={led}'
    else:
        url = f'http://{switch_ip}/offs.cgi?led={led}'
    payload = {
        "led": led,
    }
    headers = {
        "Host": switch_ip,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:56.0) Gecko/20100101 Firefox/56.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Referer": f"http://{switch_ip}/outlet.htm",
        "Authorization": "Basic c25tcDoxMjM0",
        "Connection": "keep-alive",
        "Content-Length": "0",
    }

    # print(url)
    # print(headers)
    default_string = "Could not change Lindy IP switch status, portNumber:"
    try:
        requests_status = requests.post(url, data=json.dumps(payload), headers=headers)
        requests_status.raise_for_status()
        reboot_status = ErrorCodes.SUCCESS
    except requests.exceptions.HTTPError as http_error:
        reboot_status = ErrorCodes.HTTP_ERROR
        logger.error(f"{default_string} {switch_port} status:{reboot_status} switchIP: {switch_ip} error:{http_error}")
    except requests.exceptions.ConnectionError as connection_error:
        reboot_status = ErrorCodes.CONNECTION_ERROR
        logger.error(
            f"{default_string} {switch_port} status:{reboot_status} switchIP: {switch_ip} error:{connection_error}")
    except requests.exceptions.Timeout as timeout_error:
        reboot_status = ErrorCodes.TIMEOUT_ERROR
        logger.error(
            f"{default_string} {switch_port} status:{reboot_status} switchIP: {switch_ip} error:{timeout_error}")
    except requests.exceptions.RequestException as general_error:
        reboot_status = ErrorCodes.GENERAL_ERROR
        logger.error(
            f"{default_string} {switch_port} status:{reboot_status} switchIP: {switch_ip} error:{general_error}")
    return reboot_status


def _common_switch_command(status: str, switch_ip: str, switch_port: int) -> ErrorCodes:
    """Common switch reboot rules
    :param status: ON or OFF
    :param switch_ip: ip address for the switch
    :param switch_port: port to reboot
    :return: ErrorCodes enum
    """
    port_default_cmd = 'pw%1dName=&P6%1d=%%s&P6%1d_TS=&P6%1d_TC=&' % (
        switch_port, switch_port - 1, switch_port - 1, switch_port - 1)

    cmd = 'curl --data \"'
    cmd += port_default_cmd % ("On" if status == __ON else "Off")
    cmd += '&Apply=Apply\" '
    cmd += f'http://%s/tgi/iocontrol.tgi {switch_ip}'
    cmd += '-o /dev/null '
    return _execute_command(cmd)


def _select_command_on_switch(status: str, switch_model: str, switch_port: int, switch_ip: str,
                              logger: logging.Logger) -> ErrorCodes:
    """Select the switch and execute the command
    :param status: ON or OFF
    :param switch_model: model of the switch. Supported now default and lindy
    :param switch_port: port to reboot
    :param switch_ip: ip address for the switch
    :param logger: logging.Logger obj
    :return: ErrorCodes enum, if the switch is not defined it will trow a ValueError exception
    """
    if switch_model == "default":
        return _common_switch_command(status, switch_ip, switch_port)
    elif switch_model == "lindy":
        return _lindy_switch(status, switch_port, switch_ip, logger)
    else:
        raise ValueError("Incorrect switch set to switch_model")


def reboot_machine(address: str, switch_model: str, switch_port: int, switch_ip: str, rebooting_sleep: float,
                   logger_name: str) -> typing.Tuple[ErrorCodes, ErrorCodes]:
    """Public function to reboot a machine
    :param address: Address of the machine that is being rebooted
    :param switch_model: model of the switch. Supported now default and lindy
    :param switch_port: port to reboot
    :param switch_ip: ip address for the switch
    :param rebooting_sleep: How many seconds the machine must be OFF before turn ON again
    :param logger_name: logger name defined in the main setup module
    :return: a tuple containing the outcomes of the OFF and ON commands
    """
    logger = logging.getLogger(f"{logger_name}.{__name__}")
    logger.info(f"Rebooting machine: {address}, switch IP: {switch_ip}, switch switch_port: {switch_port}")
    off_status = _select_command_on_switch(status=__OFF, switch_model=switch_model, switch_port=switch_port,
                                           switch_ip=switch_ip, logger=logger)
    time.sleep(rebooting_sleep)
    on_status = _select_command_on_switch(status=__ON, switch_model=switch_model, switch_port=switch_port,
                                          switch_ip=switch_ip, logger=logger)
    return off_status, on_status


def turn_machine_on(address: str, switch_model: str, switch_port: int, switch_ip: str, logger_name: str) -> ErrorCodes:
    """Public function to turn ON a machine
    :param address: Address of the machine that is being rebooted
    :param switch_model: model of the switch. Supported now default and lindy
    :param switch_port: port to reboot
    :param switch_ip: ip address for the switch
    :param logger_name: logger name defined in the main setup module
    :return: ErrorCodes status
    """
    logger = logging.getLogger(f"{logger_name}.{__name__}")
    logger.info(f"Turning ON machine: {address}, switch IP: {switch_ip}, switch switch_port: {switch_port}")
    return _select_command_on_switch(status=__ON, switch_model=switch_model, switch_port=switch_port,
                                     switch_ip=switch_ip, logger=logger)


if __name__ == '__main__':
    # FOR DEBUG ONLY
    # TODO: DEBUG the functions
    print("CREATING THE RebootMachine")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename="unit_test_log_RebootMachine.log",
        filemode='w'
    )
    reboot = reboot_machine(address="192.168.1.11", switch_model="lindy", switch_port=1,
                            switch_ip="192.168.1.102", rebooting_sleep=10, logger_name="REBOOT-MACHINE_LOG")

    print(f"Reboot status OFF {reboot[0]} ON {reboot[1]}")
