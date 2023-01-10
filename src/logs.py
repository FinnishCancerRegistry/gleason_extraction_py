import datetime
import logging
import os
import sys

script_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

log_dir_path = script_dir_path + "/logs/"
if not os.path.isdir(log_dir_path):
    os.makedirs(log_dir_path)

time_str = datetime.datetime.now().strftime("%Y-%m")
log_file_path = log_dir_path + time_str + '_gleasonextraction.log'
if not os.path.isfile(log_file_path):
    with open(log_file_path, "w") as f:
        f.write("")

logging.basicConfig(
	filename = log_file_path,
	level = logging.INFO,
	format='%(asctime)s	%(levelname)s	%(message)s'
)
