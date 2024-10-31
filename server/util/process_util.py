import os
import signal
import subprocess
import sys
import threading
import time

import logging

from celery.utils.log import get_task_logger
os.environ["PYTHONIOENCODING"] = "utf-8"

CACHE_POOL = {}

# 设置日志
logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

interrupt_processing_mutex = threading.RLock()
interrupt_processing = False

def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing


class HSubprocess:
    process_instance = None
    process_instance_pid = None
    environ = None
    _mswindows = False

    def __init__(self, args,environ):
        self.args = args
        self.environ=environ

    def stop(self):
        if self.process_instance is not None:

            self.process_instance.kill()
            self.process_instance = None

            try:
                try:
                    import psutil
                except ImportError:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "psutil"])
                    import psutil
                psutil.Process(self.process_instance_pid).terminate()
            except Exception as e:
                print(e)

            self.process_instance_pid = None

    def wait(self):
        interrupted = threading.Event()

        def read_stream(stream, log_func):
            while not interrupted.is_set():
                line = stream.readline()
                if not line:
                    break
                log_func(line.strip())
            stream.close()

        try:
            # Run the subprocess in the same terminal
            process = subprocess.Popen(
                self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                text=True,
                encoding="utf-8",
                env=self.environ
            )

            self.process_instance = process
            self.process_instance_pid = process.pid
            logger.info(f"Subprocess PID: {self.process_instance_pid}  \n env:{self.environ}")

            # Start threads to read stdout and stderr
            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, logging.info))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, logging.error))

            stdout_thread.start()
            stderr_thread.start()

            # Periodically check if processing is interrupted
            while process.poll() is None:
                if processing_interrupted():
                    interrupted.set()
                    process.terminate()
                    os.kill(self.process_instance_pid, signal.SIGKILL)
                    return
                time.sleep(1)  # Adjust the sleep interval as needed

            # Ensure all output is processed
            stdout_thread.join()
            stderr_thread.join()

            retcode = process.poll()
            if retcode != 0:
                raise subprocess.CalledProcessError(retcode, process.args)

        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess failed with error: pid:{self.process_instance_pid},  {e}")
            os.kill(self.process_instance_pid, signal.SIGKILL)
            raise
        except Exception as e:
            logger.error(f"An error occurred: pid:{self.process_instance_pid},  {e}")
            os.kill(self.process_instance_pid, signal.SIGKILL)
            raise
        finally:
            self.process_instance = None
            self.process_instance_pid = None
