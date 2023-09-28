"""
This module enables to setup the logger (both file and sysout)
Author(s): Amit S. Kesari
"""
import logging
from logging.handlers import RotatingFileHandler
import os, sys

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
logpath = curr_dir_path + "/logs"

if not os.path.exists(logpath):
    os.makedirs(logpath)

def get_logger(module_name):
    print(logpath)
    # setup the logger with an appropriate initial level
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    #setup log formatter
    formatter = logging.Formatter('%(asctime)s:%(levelname)s : %(name)s:(%(lineno)d) : %(message)s')

    # setup the rotating log file formatter for size 5 MB
    logFile = logpath + "/application.log"
    file_handler = RotatingFileHandler(logFile, mode='a', maxBytes=5*1024*1024, 
                                  backupCount=10, encoding=None, delay=False)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    #setup sysout handler in addition to file handler
    sysout_handler = logging.StreamHandler(sys.stdout)
    sysout_handler.setFormatter(formatter)
    sysout_handler.setLevel(logging.INFO)

    # add formatter to logger
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(sysout_handler)

    # 
    return logger