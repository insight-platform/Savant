#!/usr/bin/env python3
"""Run performance management.
TODO:
    1. Run with specified docker image, savant version
    2. Prepare 1st step: sync data
mkdir -p data
aws s3 sync --no-sign-request --endpoint-url=https://eu-central-1.linodeobjects.com s3://savant-data/demo data
OR
docker run --rm \
 -v `pwd`/data:/data \
 -e AWS_CONFIG_FILE \
 amazon/aws-cli \
 --no-sign-request \
 --endpoint-url=https://eu-central-1.linodeobjects.com \
 s3 sync s3://savant-data/demo /data
"""
import argparse
import re
import subprocess
from pathlib import Path
from typing import Generator


def launch_script(script: Path) -> Generator[str, None, None]:
    """Runs the script.

    :param script: The script.
    :return: Output rows
    """
    with subprocess.Popen(
        script,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # env=env,
        encoding='utf8',
        text=True,
    ) as process:
        while True:
            ret_code = process.poll()
            line = process.stdout.readline().strip()
            yield line
            if process.stderr:
                yield process.stderr.readline().strip()
            if ret_code is not None:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-runs', type=int, default=3, help='number of runs')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='print script output'
    )
    args = parser.parse_args()

    sample_root = Path('samples')

    fps_pattern = re.compile(r'^.*Processed \d+ frames, (?P<fps>\d+\.\d+) FPS\.$')

    for perf_script in sorted(sample_root.glob('**/run_perf.sh')):
        print('>>> Run', perf_script)

        fps_list = []
        for num in range(args.num_runs):
            for line in launch_script(perf_script):
                if not line:
                    continue

                if args.verbose:
                    print(line, '\r')

                if ' Processed ' in line:
                    match = fps_pattern.match(line)
                    if match:
                        fps_list.append(float(match['fps']))

        if fps_list:
            print(f'FPS={fps_list}, Avg={(sum(fps_list) / len(fps_list)):.2f}')
        else:
            print('Fail')


if __name__ == '__main__':
    main()
