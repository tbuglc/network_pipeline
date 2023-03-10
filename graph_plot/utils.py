import os
from pathlib import Path

def create_folder_if_not_exist(path):
    dir_exits = os.path.exists(path)
    if not dir_exits:
        os.mkdir(Path(path))