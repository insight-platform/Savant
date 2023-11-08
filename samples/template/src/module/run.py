#!/usr/bin/env python3
import os

from savant.entrypoint.main import run_module

if __name__ == '__main__':
    run_module(os.path.join(os.path.dirname(__file__), 'module.yml'))
    print('done')
