import os
from pathlib import Path

def create_folder_if_not_exist(path):
    dir_exits = os.path.exists(path)
    if not dir_exits:
        os.mkdir(Path(path))

        
def parse_output_dir(path):
    if path == '':
        raise 'Path is required'
    file_name = ''
    if '/' in path:
        file_name = path.split('/')[-1]
    if '\\' in path:
        file_name = path.split('\\')[-1]
    
    if '.' not in file_name:
        raise ValueError('file missing extension')
    
    output_dir = path.split(file_name)[0]

    return output_dir, file_name
  