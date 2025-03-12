# gpu_logging.py
import subprocess
import datetime
import time
import threading
import logging

def log_gpu_usage_periodically(interval_sec=900, logfile="gpu_usage.log"):
    """
    Logs GPU usage every `interval_sec` seconds to `logfile`.
    Runs in a daemon thread, so it won't block main script exit.
    If nvidia-smi is not available, logs N/A instead.
    """
    logger = logging.getLogger("semi_supervised_logger.gpu")

    def _gpu_logger():
        while True:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", 
                     "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                )
                usage_str = result.decode().strip()
            except Exception as e:
                usage_str = f"N/A (error: {e})"

            logger.info(f"[{timestamp}] GPU memory usage (MB): {usage_str}")

            time.sleep(interval_sec)

    t = threading.Thread(target=_gpu_logger, daemon=True)
    t.start()
