#!/usr/bin/env python3
"""Run performance management.

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
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))
from savant.utils.platform import get_platform_info
from savant.utils.version import version

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
    parser.add_argument(
        'label', nargs='+', type=str, help='run label, eg. issue number "#123".'
    )
    parser.add_argument('-n', '--num-runs', type=int, default=3, help='number of runs')
    parser.add_argument(
        '-p', '--path', type=Path, default=Path('samples'), help='path to sample(s)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='print script output'
    )
    args = parser.parse_args()

    perf_scripts = sorted(args.path.glob('**/run_perf.sh'))
    if not perf_scripts:
        sys.exit('No run_perf.sh scripts found.')

    # set of arguments to test
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

    # required arguments
    fps_period = 100
    run_args = [
        # increase time to collect batch
        'parameters.batched_push_timeout=1000',
        # short measurement period to drop the 1st and the last measurements (outliers)
        # TODO: Implement delayed start and early stop of fps measurements in GstPipeline
        f'parameters.fps_period={fps_period}',
    ]

    fps_pattern = re.compile(r'^.*Processed \d+ frames, (?P<fps>\d+\.\d+) FPS\.$')

    try:
        platform_info = get_platform_info()
    except Exception as exc:
        sys.exit(str(exc))

    dtm = datetime.now(timezone.utc)

    data = dict(
        time=dtm.isoformat(),
        savant=version.SAVANT,
        deepstream=version.DEEPSTREAM,
        labels=args.label,
        git_revision=subprocess.check_output('git rev-parse --short HEAD', shell=True)
        .decode()
        .strip(),
        platform=platform_info,
        docker=subprocess.check_output(
            'docker version --format "{{.Client.Version}}"', shell=True
        )
        .decode()
        .strip(),
        measurements=[],
    )

    logs_root = Path('logs')
    logs_root.mkdir(parents=True, exist_ok=True)
    log_file_name = f'{platform_info["nodename"]}-{dtm.strftime("%Y%m%d-%H%M%S")}'
    log_file_path = logs_root / f'{log_file_name}.log'
    json_file_path = logs_root / f'{log_file_name}.json'
    with open(log_file_path, 'w') as log_file:

        for run_cmd in run_options:
            print(f'cmd: {run_cmd}')
            log_file.write(f'cmd: {run_cmd}\n\n')

            fps_list = []
            for num in range(args.num_runs):
                log_file.write(f'run #{num}\n')

                _fps_list = []
                for line in launch_script(run_cmd + run_args):
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
                            _fps_list.append(float(match['fps']))

                if len(_fps_list) > 2:
                    _fps_list = _fps_list[1:-1]
                    num_frames = len(_fps_list) * fps_period
                    duration = sum([fps_period / fps for fps in _fps_list])
                    _fps_list = [num_frames / duration]
                fps_list.extend(_fps_list)

                log_file.write('\n')
                log_file.flush()

            if not fps_list:
                print('fail\n')
                continue

            avg_fps = sum(fps_list) / len(fps_list)
            print(f'fps: {fps_list}\navg_fps: {avg_fps:.2f}\n')

            data['measurements'].append(
                dict(
                    cmd=run_cmd,
                    # TODO: + module config
                    fps=[round(fps, 3) for fps in fps_list],
                    avg_fps=round(avg_fps, 2),
                )
            )
            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=2)


if __name__ == '__main__':
    main()
