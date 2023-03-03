import logging
import sys

# default logging level
logging_level = logging.INFO

# This will create a custom logger with the name as the module name
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
logger.setLevel(logging_level)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging_level)
formatter = logging.Formatter(
    "%(levelname)7s %(filename)s %(funcName)s() line:%(lineno)d - %(message)s"  # noqa
)
handler.setFormatter(formatter)

old_handler = logger.handlers[0]
logger.removeHandler(old_handler)
logger.addHandler(handler)
