"""Module entrypoint.
Usage examples:

> echo "
name: test-module
parameters:
  batch_size: 1
" | python -m savant.entrypoint

> python -m savant.entrypoint -e some/module/config.yml

> cat some/module/config.yml | python -m savant.entrypoint
"""
import argparse
import sys

from savant.entrypoint.main import build_module_engines, main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m savant.entrypoint')
    parser.add_argument(
        '-e',
        '--build-engines-only',
        action='store_true',
        help='builds module model\'s engines and exit',
    )
    parser.add_argument(
        'config',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='config file to read, if empty, STDIN is used',
    )
    args = parser.parse_args()

    if args.config.isatty():
        parser.print_help()
        exit(0)

    if args.build_engines_only:
        build_module_engines(args.config)
    else:
        main(args.config)
