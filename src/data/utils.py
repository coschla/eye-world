from pathlib import Path


def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue_1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    path = Path(fname_path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    new_path = parent / f"{stem}_{i}{suffix}"
    while new_path.exists():
        i += 1
        new_path = parent / f"{stem}_{i}{suffix}"

    return new_path


def get_nonexistant_shard_path(fname_path_template):
    """
    Get the index for a shard path that does not exist by incrementing.

    Assumes `fname_path_template` is a format string like: "shard_%d.txt"

    Examples
    --------
    >>> get_nonexistant_shard_path('output_%d.txt')
    0  # if output_0.txt does not exist
    >>> get_nonexistant_shard_path('output_%d.txt')
    5  # if output_0.txt to output_4.txt exist
    """
    path = Path(fname_path_template % 0)
    if not path.is_file():
        return 0

    index = 1
    while Path(fname_path_template % index).exists():
        index += 1

    return index
