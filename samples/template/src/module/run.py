#!/usr/bin/env python3
import os

from savant.entrypoint.main import main

if __name__ == '__main__':
    main(os.path.join(os.path.dirname(__file__), 'module.yml'))
    print('done')
