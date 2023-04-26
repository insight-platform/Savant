import platform
import subprocess


def is_aarch64():
    """Checks if the current platform is Jetson."""
    return platform.machine() == 'aarch64'


def get_l4t_version():
    """Return L4T version (Jetson).
    Eg. Xavier NX L4T 35.1.0 => 35.1.0-20220810203728 => [35, 1, 0]
    or Jetson Nano L4T 32.6.1 => [32, 6, 1]
    """
    output = subprocess.check_output(
        ['dpkg-query', '--showformat=${Version}', '--show', 'nvidia-l4t-core']
    )
    version = output.decode().split('-', 1)[0]
    return list(map(int, version.split('.')))
