import logging
import time
from multiprocessing import Lock

from colorama import Fore

import random
import numpy as np
import torch

SEED = 32


class _Pipe:
    def __init__(self):
        self.content = None

    def send(self, item):
        self.content = item

    def recv(self):
        return self.content


class StopWatch:
    def __init__(self):
        self.start_instance = time.time()

    def stop(self) -> float:
        current_instance = time.time()
        return current_instance - self.start_instance


def hhmmss(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


stdout_lock = Lock()


def sprint(status: str, message: str):
    """
    This method logs a message with different levels (debug, info, warning, error, critical).

    :param status: (str) a level of the 5 levels
    :param message: the message yp be printed.
    :return:
    """

    status = status.upper()
    with stdout_lock:
        logging.basicConfig(level=logging.DEBUG)
        if status == 'INFO':
            # Green
            logging.log(20, Fore.GREEN + message + Fore.WHITE)
        elif status == 'WARNING':
            # Yellow
            logging.log(30, Fore.YELLOW + message + Fore.WHITE)
        elif status == 'ERROR':
            # Red
            logging.log(40, Fore.RED + message + Fore.WHITE)
        elif status == 'CRITICAL':
            # Black
            logging.log(40, Fore.BLACK + message + Fore.WHITE)
        elif status == 'DEBUG':
            # Blue
            logging.log(10, Fore.BLUE + message + Fore.WHITE)
        else:
            sprint('ERROR', 'ERROR: Invalid Status')

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
