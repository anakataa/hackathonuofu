"""Wrapper around the default logging library"""

import logging
import os
from datetime import datetime

from hack import const


def new_log_file() -> str:
    """Generate a unique filename for a log file. Ensure that the path exists.
    Filenames produced by this function contain an ISO 8601 timestamp.

    This function must always succeed or raise an exception otherwise.

    Returns:
        str: The path to the new log file.
    """

    # Create the logs directory if it doesn't already exist.
    os.makedirs(const.logs_dir_path, exist_ok=True)

    # Generate the filename from the current time.
    log_name: str = f"{datetime.now().isoformat(sep="_")}{const.log_ext}"

    # Get the absolute path to the log file.
    log_path: str = os.path.join(const.logs_dir_path, log_name)

    return log_path


class IgnoreAllFilter(logging.Filter):

    def filter(self, record):
        return False


# The logger is created and managed by gunicorn.
log: logging.Logger = logging.getLogger("gunicorn.error")
