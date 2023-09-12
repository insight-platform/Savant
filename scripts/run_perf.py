#!/usr/bin/env python3
"""Run performance management.

TODO:
Run context
* platform
  * os
  * cpu
  * gpu
  * ram?
  * mode (Jetson: cores, clocks)
* version
  * VERSION
  * revision (`git rev-parse HEAD`)
  * issue? (#NNN)
  * [label]
* date+time (+0, `datetime.now(timezone.utc)`)
* parameters (buffer_queues, batch_size, multistream/uridecodebin, num_streams)
* sample name - dir name in samples/
* module config

TODO: Sync data step
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
import itertools
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Optional, Union

# ANSI codes are used, for example, to colorize terminal output. They come from savant_rs logging
# ANSI codes interfere with parsing FPS from the output.
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def launch_script(
    args: Union[str, list], env: Optional[Dict[str, str]] = None
) -> Generator[str, None, None]:
    """Runs the script.

    :param args: A string, or a sequence of script arguments.
    :param env: The environment variables for the new process.
    :return: Output rows
    """
    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
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
    perf_scripts = sorted(sample_root.glob('**/run_perf.sh'))

    run_options = [
        perf_scripts,
        # uridecodebin, multistream=1, multistream=2, etc.
        [0, 1, 2, 4, 8],
        [
            'parameters.batch_size=1',
            'parameters.batch_size=4',
            'parameters.batch_size=8',
        ],
        ['parameters.buffer_queues=null', 'parameters.buffer_queues.length=10'],
    ]

    # options combinations where batch_size >= number-of-streams
    run_options = [
        [str(opt) for opt in opts]
        for opts in itertools.product(*run_options)
        if int(opts[2].replace('parameters.batch_size=', '')) >= opts[1]
    ]

    # TODO: Fix yolov5nface to be able to build engine with batch-size > 1
    # remove face detector samples with batch-size > 1
    run_options = [
        opts
        for opts in run_options
        if not ('age_gender_recognition' in opts[0] or 'face_reid' in opts[0])
        or int(opts[2].replace('parameters.batch_size=', '')) == 1
    ]

    fps_pattern = re.compile(r'^.*Processed \d+ frames, (?P<fps>\d+\.\d+) FPS\.$')

    logs_root = Path('logs')
    log_file_path = logs_root / (
        datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '.log'
    )
    with open(log_file_path, 'w') as log_file:

        for run_cmd in run_options:
            print('>>> Run', run_cmd)
            log_file.write(f'CMD: {run_cmd}\n')

            fps_list = []
            for num in range(args.num_runs):
                log_file.write(f'\nRUN #{num}\n')

                for line in launch_script(run_cmd):
                    if not line:
                        continue

                    log_file.write(f'{line}\n')
                    if args.verbose:
                        print(line, '\r')

                    line = ANSI_ESCAPE_PATTERN.sub('', line)
                    # this check won't work if there are ANSI codes in the line
                    if ' Processed ' in line:
                        match = fps_pattern.match(line)
                        if match:
                            fps_list.append(float(match['fps']))

                log_file.flush()

            if fps_list:
                print(f'FPS={fps_list}, Avg={(sum(fps_list) / len(fps_list)):.2f}')
            else:
                print('Fail')


if __name__ == '__main__':
    main()
