import os
from pathlib import Path

def create_folder_if_not_exist(input_dir):
    dir_exits = os.input_dir.exists(input_dir)
    if not dir_exits:
        os.mkdir(Path(input_dir))