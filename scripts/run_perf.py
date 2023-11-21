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
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parent.parent))
from savant.utils.platform import get_jetson_stats, get_platform_info
from savant.utils.version import version

# ANSI codes are used, for example, to colorize terminal output.
# They come from savant_rs logging.
# ANSI codes interfere with parsing FPS from the output.
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
FPS_PATTERN = re.compile(
    r'^.*Processed (?P<num_frames>\d+) frames, (?P<fps>\d+\.\d+) FPS\.$'
)
STATS_PATTERN = re.compile(
    r'^(?P<name>\w+): min=(?P<min>\d+), max=(?P<max>\d+), avg=(?P<avg>\d+\.\d+)$'
)

# short measurement period to drop the 1st and the last measurements (outliers)
# TODO: Implement delayed start and early stop of fps measurements in pipeline
FPS_PERIOD = 100


def process_logs_file(logs_file_path: Path, stats_vars: List[str]) -> Dict[str, Any]:
    """Get stats from log file."""
    stats = {k: {} for k in stats_vars}
    fps_list = []
    with open(logs_file_path, mode='r') as log_file:
        for line in log_file:
            line = line.rstrip()
            if not line:
                continue

            # checks won't work if there are ANSI codes in the line
            line = ANSI_ESCAPE_PATTERN.sub('', line)

            if ' Processed ' in line:
                match = FPS_PATTERN.match(line)
                if match:
                    fps_list.append((int(match['num_frames']), float(match['fps'])))
                continue

            for stats_var in stats_vars:
                if stats_var in line:
                    match = STATS_PATTERN.match(line)
                    if match:
                        stats[match['name']]['min'] = int(match['min'])
                        stats[match['name']]['max'] = int(match['max'])
                        stats[match['name']]['avg'] = float(match['avg'])
                    break

    # remove perhaps outliers
    if len(fps_list) > 2:
        fps_list = fps_list[1:-1]
    # average
    if len(fps_list) > 1:
        num_frames = sum([num for num, _ in fps_list])
        duration = sum({num / fps for num, fps in fps_list})
        fps_list = [(num_frames, num_frames / duration)]
    if fps_list:
        stats['fps'] = {'avg': fps_list[0][1]}

    return stats


def convert_run_cmd_to_filename(run_cmd: List[str]) -> str:
    """Converts run_cmd args to valid file name."""
    # use sample dir name as sample name,
    # e.g. "samples/opencv_cuda_bg_remover_mog2/run_perf.sh"
    # gives "opencv_cuda_bg_remover_mog2"
    sample_name = run_cmd[0].split('/')[1]
    # concat with other args
    filename = sample_name + '-' + ''.join(run_cmd[1:])
    filename = filename.lower()
    # sanitize
    # filename = re.sub(r'[^\w\s-]', '', filename.lower())
    return re.sub(r'[-\s]+', '-', filename).strip('-_')


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'label', nargs='+', type=str, help='run label, eg. issue number "#123".'
    )
    parser.add_argument('-n', '--num-runs', type=int, default=5, help='number of runs')
    parser.add_argument(
        '-p', '--path', type=Path, default=Path('samples'), help='path to sample(s)'
    )
    parser.add_argument(
        '-m', '--multi-option', action='store_true', help='multi-option launch'
    )
    parser.add_argument(
        '-s',
        '--stats',
        action='store_true',
        help='add stat_logger to count frames and objects',
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
        # TODO: Check nvp model
        # nvp_model = 'MODE_20W_6CORE'  # Xavier NX
        # if jetson_stats['nvp model'] != nvp_model:
        #     sys.exit(f'nvp model should be set in {nvp_model}')
        if jetson_stats['jetson_clocks'] != 'ON':
            sys.exit('jetson_clocks should be ON.')

    # collect perf scripts
    perf_scripts = sorted(str(path) for path in args.path.glob('**/run_perf.sh'))
    if not perf_scripts:
        sys.exit('No run_perf.sh scripts found.')

    if args.multi_option:
        # source + batch combinations
        # where source: uridecodebin=0, multistream=1, multistream=2, etc.
        source_batch_options = [
            # num_streams=1, batch_size=1,4
            # (nvstreammux doesn't collect batch size > 4 with one source)
            ('0', '.parameters.batch_size=1'),
            ('0', '.parameters.batch_size=4'),
            ('1', '.parameters.batch_size=1'),
            ('1', '.parameters.batch_size=4'),
            # num_streams=4, batch_size=4
            ('4', '.parameters.batch_size=4'),
        ]
        if not is_jetson:
            source_batch_options += [
                # num_streams=4, batch_size=8
                ('4', '.parameters.batch_size=8'),
                # num_streams=8, batch_size=8
                ('8', '.parameters.batch_size=8'),
            ]
        queue_options = [
            '.parameters.buffer_queues=null',
            '.parameters.buffer_queues.length=10',
        ]
        run_options = [perf_scripts, source_batch_options, queue_options]
        run_options = [
            [opts[0], opts[1][0], opts[1][1], opts[2]]
            for opts in itertools.product(*run_options)
        ]

    # default: uridecodebin
    else:
        run_options = [[perf_script, '0'] for perf_script in perf_scripts]

    # required arguments
    run_args = [
        # increase time to collect batch
        '.parameters.batched_push_timeout=40000',
        # correct fps_period
        f'.parameters.fps_period={FPS_PERIOD}',
    ]
    if args.stats:
        run_args += [
            '.pipeline.elements += {'
            '"element": "pyfunc", '
            '"module": "savant.utils.stat_logger", '
            '"class_name": "StatLogger"'
            '}'
        ]

    stats_vars = []
    if args.stats:
        stats_vars += [
            'num_frames_in_batch',
            'num_frames_per_source',
            'num_objects_per_source',
        ]

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
    run_name = f'{platform_info["nodename"]}-{dtm.strftime("%Y%m%d-%H%M%S")}'
    if args.multi_option:
        run_name += '-multi'
    json_file_path = logs_root / f'{run_name}.json'

    logs_dir = logs_root / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    for run_cmd in run_options:
        logs_file_name = convert_run_cmd_to_filename(run_cmd)
        print(f'cmd: {run_cmd}')
        stats = []
        for num in range(args.num_runs):
            logs_file_path = logs_dir / f'{logs_file_name}-{num:02d}.log'
            print(f'run #{num}')
            # save logs
            with open(logs_file_path, 'w') as log_file:
                log_file.write(f'cmd: {run_cmd}\n')
                log_file.write(f'run #{num}\n')
                # if is_jetson:
                #     log_file.write(f'stats: {get_jetson_stats()}\n')
                log_file.flush()
                subprocess.run(run_cmd + run_args, stdout=log_file, stderr=log_file)
            # process logs
            _stats = process_logs_file(logs_file_path, stats_vars)
            if 'fps' not in _stats:
                print('fail')
                continue
            print(f'stats\n{json.dumps(_stats, indent=2)}')
            stats.append(_stats)

        # results
        fps_list = [stat['fps']['avg'] for stat in stats]
        if not fps_list:
            print('fail')
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
            for val in ('min', 'max', 'avg'):
                val_list = [stat[stats_var][val] for stat in stats]
                print(f'{stats_var}_{val}: {val_list}')
                measurement[f'{stats_var}_{val}'] = val_list

        data['measurements'].append(measurement)
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2)

        print()


if __name__ == '__main__':
    main()
