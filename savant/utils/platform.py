"""Platform utils."""

import os
import platform
import subprocess
from functools import lru_cache


def is_aarch64() -> bool:
    """Checks if the current platform is Jetson."""
    return platform.machine() == 'aarch64'


def get_l4t_version() -> list:
    """Returns L4T version (Jetson).
    Eg. Xavier NX L4T 35.1.0 => 35.1.0-20220810203728 => [35, 1, 0]
    or Jetson Nano L4T 32.6.1 => [32, 6, 1]
    """
    output = subprocess.check_output(
        ['dpkg-query', '--showformat=${Version}', '--show', 'nvidia-l4t-core']
    )
    version = output.decode().split('-', 1)[0]
    return list(map(int, version.split('.')))


class UnsupportedPlatform(Exception):
    """UnsupportedPlatform exception class."""


def get_os_description() -> str:
    """Returns a human-readable OS description.

    Uses the lsb_release python module when it is available, otherwise falls
    back to PRETTY_NAME from /etc/os-release.
    """
    try:
        import lsb_release

        return lsb_release.get_os_release()['DESCRIPTION']
    except ImportError:
        pass

    try:
        with open('/etc/os-release', encoding='utf-8') as os_release:
            for line in os_release:
                key, _, value = line.partition('=')
                if key == 'PRETTY_NAME':
                    return value.strip().strip('"')
    except OSError:
        pass

    return 'Unknown'


@lru_cache(maxsize=1)
def get_platform_info() -> dict:
    """Returns current platform info."""
    uname_result = os.uname()
    platform_info = dict(
        nodename=uname_result.nodename,
        machine=uname_result.machine,
        sysname=uname_result.sysname,
        release=uname_result.release,
    )

    if platform_info['sysname'] != 'Linux':
        raise UnsupportedPlatform(f'Unsupported platform {platform_info["sysname"]}.')

    platform_info['os'] = get_os_description()

    if platform_info['machine'] == 'x86_64':
        device, driver = (
            subprocess.check_output(
                [
                    'nvidia-smi',
                    '--id=0',
                    '--query-gpu=gpu_name,driver_version',
                    '--format=csv,noheader',
                ]
            )
            .decode()
            .strip()
            .split(', ')
        )
        platform_info['dgpu'] = dict(device=device, driver=driver)

    elif platform_info['machine'] == 'aarch64':
        try:
            import jtop

            jetson = jtop.jtop()
            jetson.start()
            platform_info['jetson'] = dict(
                model=jetson.board['hardware']['Model'],
                l4t=jetson.board['hardware']['L4T'],
                jetpack=jetson.board['hardware']['Jetpack'],
                cuda=jetson.board['libraries']['CUDA'],
                tensorrt=jetson.board['libraries']['TensorRT'],
            )
            jetson.close()
        except ImportError:
            platform_info['jetson'] = dict(l4t='.'.join(get_l4t_version()))

    else:
        raise UnsupportedPlatform(f'Unsupported platform {platform_info["machine"]}.')

    cpu_info = subprocess.check_output('cat /proc/cpuinfo', shell=True).decode().strip()
    for line in cpu_info.split('\n'):
        if 'model name' in line:
            platform_info['cpu'] = line.split(': ', 2)[1]
            break

    return platform_info


def get_jetson_stats():
    """Returns a simplified version of tegrastats."""
    import jtop

    jetson = jtop.jtop()
    jetson.start()
    stats = jetson.stats
    jetson.close()

    return stats
