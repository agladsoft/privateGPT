import os
import time
import psutil
import inspect
import logging
import nvidia_smi
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Callable
from typing import Optional
from functools import wraps
from multiprocessing import Event
from multiprocessing import Value
from multiprocessing import Process
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as EventClass


def get_stream_handler():
    stream_handler: logging.StreamHandler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    return stream_handler


logger: logging.getLogger = logging.getLogger(__name__)
logger.addHandler(get_stream_handler())
logger.setLevel(logging.INFO)


def _memory_monitor(pid: int, stop_event: EventClass,
                    max_ram_pid: Synchronized,
                    max_ram: Synchronized,
                    max_vram: Synchronized,
                    total_ram: Synchronized,
                    total_vram: Synchronized) -> None:
    """Monitor Memory consumption in parallel to a given process (RAM and VRAM).
    Args:
        pid: ID of the Process to monitor
        stop_event: Shared event triggered when the monitoring has to stop
        max_ram_pid: Peak number of bytes used by the profiled process in the RAM
        max_ram: Peak number of bytes used in the RAM
        max_vram: Peak number of bytes used in the Video RAM (GPU)
        total_ram: Total number of bytes available in RAM (used+free)
        total_vram: Total number of bytes available in VRAM (used+free)
    """
    max_ram_pid_value = 0
    max_ram_value = 0
    max_vram_value = 0

    nvidia_smi.nvmlInit()
    assert nvidia_smi.nvmlDeviceGetCount() == 1  # Assume a single GPU
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    total_ram.value = psutil.virtual_memory().total
    total_vram.value = int(nvidia_smi.nvmlDeviceGetMemoryInfo(handle).total)

    # while not stop_event.is_set():
    max_ram_pid_value = max(max_ram_pid_value, psutil.Process(pid).memory_info().rss)
    max_ram_value = max(max_ram_value, psutil.virtual_memory().available)
    gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    max_vram_value = max(max_vram_value, gpu_info.used)
    time.sleep(0.2)  # Arbitrary time interval (in s)

    max_ram_pid.value = max_ram_pid_value
    max_ram.value = max_ram_value
    max_vram.value = max_vram_value
    nvidia_smi.nvmlShutdown()


def get_function_name(func: Callable) -> str:
    """Get file and function name."""
    module_name = func.__module__.split('.')[-1]
    if module_name == "__main__":
        module_name = os.path.splitext(os.path.basename(inspect.getfile(func)))[0]
    return f"{module_name}::{func.__name__}"


FuncT = TypeVar('FuncT', bound=Callable[..., Any])


def memory_peak_profile(func: FuncT) -> FuncT:
    """Memory peak Profiling decorator (Both RAM and VRAM)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryScope(get_function_name(func)):
            retval = func(*args, **kwargs)
        return retval

    return cast(FuncT, wrapper)


class MemoryScope:
    """Use as context `with MemoryScope(name):`."""

    def __init__(self, name: str):
        self._name = name
        self._max_ram_process: Synchronized = Value("L", 0)
        self._max_ram: Synchronized = Value("L", 0)
        self._max_vram: Synchronized = Value("L", 0)
        self._total_ram: Synchronized = Value("L", 0)
        self._total_vram: Synchronized = Value("L", 0)

        self._stop_event = Event()
        self._monitor_process: Optional[Process] = None

    def __enter__(self):
        pid = os.getpid()
        self._monitor_process = Process(target=_memory_monitor,
                                        args=(pid,
                                              self._stop_event,
                                              self._max_ram_process,
                                              self._max_ram,
                                              self._max_vram,
                                              self._total_ram,
                                              self._total_vram))
        self._monitor_process.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._stop_event.set()  # Signal the monitor process to stop
        assert self._monitor_process is not None
        self._monitor_process.join()

        max_ram = bytes2human(int(self._max_ram.value))
        max_ram_pid = bytes2human(int(self._max_ram_process.value))
        tot_ram = bytes2human(int(self._total_ram.value))
        max_vram = bytes2human(int(self._max_vram.value))
        tot_vram = bytes2human(int(self._total_vram.value))

        logger.info(f"RAM-Peak = {max_ram}/{tot_ram} ({max_ram_pid}) "
                    f"/ VRAM-Peak = {max_vram}/{tot_vram}")


def bytes2human(number: int, decimal_unit: bool = True) -> str:
    """Convert number of bytes in a human readable string.
    Args:
        number (int): Number of bytes
        decimal_unit (bool): If specified, use 1 kB (kilobyte)=10^3 bytes.
            Otherwise, use 1 KiB (kibibyte)=1024 bytes
    Returns:
        str: Bytes converted in readable string
    """
    symbols = ['K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    symbol_values = [(symbol,
                      1000**(i+1) if decimal_unit else (1 << (i+1) * 10))
                     for i, symbol in enumerate(symbols)]

    for symbol, value in reversed(symbol_values):
        if number >= value:
            suffix = "B" if decimal_unit else "iB"
            return f"{float(number)/value:.2f}{symbol}{suffix}"

    return f"{number} B"


if __name__ == "__main__":

    @memory_peak_profile
    def summ(x, y):
        return x ** y

    summ(5, 1000)
