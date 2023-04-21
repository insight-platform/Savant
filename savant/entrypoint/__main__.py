"""Module entrypoint.

>>> python -m savant.entrypoint {config_file_path}
"""
import sys
from main import main

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        print('Module config file path is expected as a CLI argument.')
