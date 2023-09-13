"""Platform utils."""
import os
import platform
import subprocess
from functools import lru_cache


def is_aarch64():
    """Checks if the current platform is Jetson."""
    return platform.machine() == 'aarch64'


def get_l4t_version():
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

    import lsb_release

    platform_info['os'] = lsb_release.get_os_release()['DESCRIPTION']

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
        platform_info['jetson'] = dict()
        # TODO:
        # Jetson> sudo python3
        # >>> import jtop
        # >>> with jtop.jtop() as jetson:
        # ...     jetson.board
        # {
        # 'hardware': {'Model': 'NVIDIA Jetson Xavier NX Developer Kit', '699-level Part Number': '699-13668-0000-300 B.0', 'P-Number': 'p3668-0000', 'Module': 'NVIDIA Jetson Xavier NX (Developer kit)', 'SoC': 'tegra194', 'CUDA Arch BIN': '7.2', 'Codename': 'Jakku', 'Serial Number': '1421321065967', 'L4T': '35.4.1', 'Jetpack': ''},
        # 'platform': {'Machine': 'aarch64', 'System': 'Linux', 'Distribution': 'Ubuntu 20.04 focal', 'Release': '5.10.120-tegra', 'Python': '3.8.10'},
        # 'libraries': {'CUDA': '11.4.315', 'OpenCV': '4.5.4', 'OpenCV-Cuda': False, 'cuDNN': '8.6.0.166', 'TensorRT': '8.5.2.2', 'VPI': '2.3.9', 'Vulkan': '1.3.204'}
        # }

    else:
        raise UnsupportedPlatform(f'Unsupported platform {platform_info["machine"]}.')

    cpu_info = subprocess.check_output('cat /proc/cpuinfo', shell=True).decode().strip()
    for line in cpu_info.split('\n'):
        if 'model name' in line:
            platform_info['cpu'] = line.split(': ', 2)[1]
            break

    return platform_info
