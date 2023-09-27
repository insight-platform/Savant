#!/usr/bin/env python3
"""Run performance management.

TODO: Add module config to measurement context.

TODO: Sync data step
mkdir -p data
aws s3 sync --no-sign-request --endpoint-url=https://eu-central-1.linodeobjects.com \
 s3://savant-data/demo data
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
from savant.utils.platform import get_jetson_stats, get_platform_info
from savant.utils.version import version

# ANSI codes are used, for example, to colorize terminal output.
# They come from savant_rs logging.
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
    """Main."""
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

    # check platform
    try:
        platform_info = get_platform_info()
    except Exception as exc:
        sys.exit(str(exc))

    is_jetson = platform_info['machine'] == 'aarch64'

    if is_jetson:
        jetson_stats = get_jetson_stats()
        # TODO: Support check for other Jetsons, currently Xavier NX specific
        nvp_model = 'MODE_20W_6CORE'
        if jetson_stats['nvp model'] != nvp_model:
            sys.exit(f'nvp model should be set in {nvp_model}')
        if jetson_stats['jetson_clocks'] != 'ON':
            sys.exit('jetson_clocks should be ON.')

    # check perf scripts
    perf_scripts = sorted(args.path.glob('**/run_perf.sh'))
    if not perf_scripts:
        sys.exit('No run_perf.sh scripts found.')

    # uridecodebin, multistream=1, multistream=2, etc.
    stream_options = [0, 1, 2, 4] if is_jetson else [0, 1, 2, 4, 8]
    batch_options = [1, 4] if is_jetson else [1, 4, 8]

    # set of arguments to test
    run_options = [
        perf_scripts,
        stream_options,
        [f'parameters.batch_size={batch_size}' for batch_size in batch_options],
        ['parameters.buffer_queues=null', 'parameters.buffer_queues.length=10'],
    ]

    run_options = [
        [str(opt) for opt in opts]
        for opts in itertools.product(*run_options)
        # only combinations where batch_size >= number-of-streams
        if int(opts[2].replace('parameters.batch_size=', '')) >= opts[1]
    ]

    # required arguments
    fps_period = 100
    run_args = [
        # increase time to collect batch
        'parameters.batched_push_timeout=40000',
        # short measurement period to drop the 1st and the last measurements (outliers)
        # TODO: Implement delayed start and early stop of fps measurements in pipeline
        f'parameters.fps_period={fps_period}',
    ]

    fps_pattern = re.compile(r'^.*Processed \d+ frames, (?P<fps>\d+\.\d+) FPS\.$')
    stats_pattern = re.compile(
        r'^(?P<name>\w+): min=(?P<min>\d+), max=(?P<max>\d+), avg=(?P<avg>\d+\.\d+)$'
    )
    stats_vars = {
        'num_frames_in_batch',
        'num_frames_per_source',
        'num_objects_per_source',
    }

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
    with open(log_file_path, mode='w', encoding='utf-8') as log_file:

        for run_cmd in run_options:
            print(f'cmd: {run_cmd}')
            log_file.write(f'cmd: {run_cmd}\n\n')

            fps_list = []
            stats = {k: {'min': [], 'max': [], 'avg': []} for k in stats_vars}
            for num in range(args.num_runs):
                log_file.write(f'run #{num}\n')

                if is_jetson:
                    log_file.write(f'stats: {get_jetson_stats()}\n')

                _fps_list = []
                for line in launch_script(run_cmd + run_args):
                    if not line:
                        continue

                    log_file.write(f'{line}\n')
                    if args.verbose:
                        print(line, '\r')

                    # checks won't work if there are ANSI codes in the line
                    line = ANSI_ESCAPE_PATTERN.sub('', line)
                    if ' Processed ' in line:
                        match = fps_pattern.match(line)
                        if match:
                            _fps_list.append(float(match['fps']))
                    for stats_var in stats_vars:
                        if stats_var in line:
                            match = stats_pattern.match(line)
                            if match:
                                stats[match['name']]['min'].append(int(match['min']))
                                stats[match['name']]['max'].append(int(match['max']))
                                stats[match['name']]['avg'].append(
                                    round(float(match['avg']), 2)
                                )
                            break

                if len(_fps_list) > 2:
                    _fps_list = _fps_list[1:-1]
                    num_frames = len(_fps_list) * fps_period
                    duration = sum({fps_period / fps for fps in _fps_list})
                    _fps_list = [num_frames / duration]
                fps_list.extend(_fps_list)

                log_file.write('\n')
                log_file.flush()

            if not fps_list:
                print('fail\n')
                continue

            fps = sum(fps_list) / len(fps_list)
            fps_list = [round(fps, 2) for fps in fps_list]
            print(f'fps: {fps:.2f}\nfps_avg: {fps_list}')

            measurement = dict(
                cmd=run_cmd,
                fps_avg=fps_list,
                fps=round(fps, 2),
            )

            for stats_var in stats_vars:
                for val in {'min', 'max', 'avg'}:
                    print(f'{stats_var}_{val}: {stats[stats_var][val]}')
                    measurement[f'{stats_var}_{val}'] = stats[stats_var][val]

            data['measurements'].append(measurement)
            with open(json_file_path, mode='w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2)

            print('\n')


if __name__ == '__main__':
    main()
