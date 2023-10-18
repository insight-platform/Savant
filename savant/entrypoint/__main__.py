"""Module entrypoint.

>>> python -m savant.entrypoint {config_file_path} [{dot_list_yaml_config_args}]

TODO: Add configuration support from STDIN to be able
    to create configuration on the fly using, for example, yq
    `if sys.argv[1] == '-': ...`
"""
import sys

from savant.entrypoint.main import main

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(*sys.argv[1:])
    else:
        print('Module config file path is expected as a CLI argument.')
