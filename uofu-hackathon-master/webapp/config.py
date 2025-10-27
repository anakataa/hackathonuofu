"""This configuration file is read by Gunicorn on startup."""

import logging
import multiprocessing
import sys
from typing import Any

from hack import const, logger
from hack.logger import log

# from hack import auth, const, inventory, logger, word_list
# from hack.logger import log
# from hack.origins import utils as origins


def init() -> None:
    """Initialize the app before launching worker threads."""

    pass


from gunicorn.arbiter import Arbiter
from gunicorn.http.message import Request
from gunicorn.http.wsgi import Response
from gunicorn.workers.base import Worker


# https://docs.gunicorn.org/en/latest/settings.html#settings
# ------------------------------------------------------------------------------

# The module path to the WSGI application.
# https://docs.gunicorn.org/en/latest/settings.html#wsgi-app
wsgi_app: str = "hack.flaskapp:app"

# Restart workers when the code changes.
# https://docs.gunicorn.org/en/latest/settings.html#reload
reload: bool = False

# The log file to write to for logging access messages.
# https://docs.gunicorn.org/en/latest/settings.html#accesslog
# "-" writes to stdout.
accesslog: str = "-"

# The log file to write to for logging error messages.
# https://docs.gunicorn.org/en/latest/settings.html#errorlog
# "-" writes to stderr.
errorlog: str = logger.new_log_file()

# The least severe messages that will be displayed.
# https://docs.gunicorn.org/en/latest/settings.html#loglevel
# debug, info, warning, error, critical
loglevel: str = "debug"

# https://docs.gunicorn.org/en/latest/settings.html#capture-output
# True  - Redirect stdout/stderr to the file in errorlog.
# False - Do not redirect output.
capture_output: bool = True


# This method is executed immediately before the Gunicorn master process is initialized.
# This method is only executed once, regardless of the number of workers.
# Defining this method overrides the default on_starting handler for Gunicorn.
# https://docs.gunicorn.org/en/stable/settings.html#on-starting
def on_starting(arbiter: Arbiter) -> None:
    # Disable the default Gunicorn access messages.
    logging.getLogger("gunicorn.access").addFilter(logger.IgnoreAllFilter())

    # Initialize the app before launching the worker threads
    init()


# This method is executed before each request is processed.
# Defining this method overrides the default pre_request handler for Gunicorn.
# https://docs.gunicorn.org/en/stable/settings.html#pre-request
def pre_request(worker: Worker, req: Request) -> None:
    pass


# This method is executed after each request has been processed and a response generated.
# Defining this method overrides the default post_request handler for Gunicorn.
# https://docs.gunicorn.org/en/stable/settings.html#post-request
def post_request(
    worker: Worker, req: Request, environ: dict[str, Any], res: Response
) -> None:
    worker.log.info(
        str([req.remote_addr, req.method, res.status, req.path, req.headers])
    )


# Run Gunicorn as a service.
# https://docs.gunicorn.org/en/latest/settings.html#daemon
daemon: bool = False

# # The user to run Gunicorn as.
# # https://docs.gunicorn.org/en/latest/settings.html#user
# user = "custom_name"

# # The group to run Gunicorn as.
# # https://docs.gunicorn.org/en/latest/settings.html#group
# group = "custom_name"

# The address and port to listen for requests on.
# https://docs.gunicorn.org/en/latest/settings.html#bind
# 127.0.0.1 or localhost block outside connections.
# 0.0.0.0 permits outside connections (dangerous!).
# 80 is the default port for HTTP.
# 443 is the default port for HTTPS.
# Listening on ports below 1024 requires root privileges.
bind: str = "0.0.0.0:5000"

# The Gunicorn docs recommend launching twice the number of workers as cores.
# https://docs.gunicorn.org/en/latest/design.html#how-many-workers
# https://docs.gunicorn.org/en/latest/settings.html#workers
workers: int = multiprocessing.cpu_count() * 2

# The number of threads per workers.
# https://docs.gunicorn.org/en/latest/design.html#threads
threads: int = 1  # threads per worker

# ------------------------------------------------------------------------------


if __name__ != "__config__":
    # The debug server is running instead of Gunicorn
    from hack.flaskapp import app

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(loglevel.upper())
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.setLevel(handler.level)
    log.addHandler(handler)

    init()
