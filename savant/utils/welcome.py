from savant.utils.version import version


def get_welcome_message() -> str:
    """Returns the welcome message that is used
    when starting Savant container.
    """
    version_info = (
        f'Savant Version {version.SAVANT}\n'
        f'Savant RS Version {version.SAVANT_RS}\n'
        f'DeepStream Version {version.DEEPSTREAM}'
    )
    return f'\n============\n== Savant ==\n============\n\n{version_info}\n'


def get_starting_message(name: str):
    """Returns the message that is used
    when starting Savant component (module, adapter).
    """
    version_info = (
        f'Savant {version.SAVANT}, '
        f'Savant RS {version.SAVANT_RS}, '
        f'DeepStream {version.DEEPSTREAM}'
    )
    return f'Starting the {name}. Packages version info: {version_info}.'


if __name__ == '__main__':
    print(get_welcome_message())
