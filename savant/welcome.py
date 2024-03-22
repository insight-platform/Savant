from savant.utils.version import version


def get_welcome_message() -> str:
    return (
        '\n============\n== Savant ==\n============\n\n'
        + '\n'.join(version.get_list())
        + '\n'
    )


if __name__ == '__main__':
    print(get_welcome_message())
