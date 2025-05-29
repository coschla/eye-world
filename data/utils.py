import re
from datetime import datetime
from pathlib import Path


def sort_files_by_timestamp(directory):
    """
    Sorts .txt files in the directory by the timestamp in their filename.

    Expected filename format:
    e.g., 71_RZ_2901714_Aug-16-11-54-21.txt

    Args:
        directory (str or Path): Path to the directory containing the files.

    Returns:
        List[Path]: Sorted list of file paths by datetime.
    """
    directory = Path(directory)
    txt_files = list(directory.glob("*.txt"))

    def extract_datetime(file_path):
        match = re.search(r"_(\w{3}-\d{2}-\d{2}-\d{2}-\d{2})", file_path.stem)
        if match:
            try:
                # Convert to datetime object
                return datetime.strptime(match.group(1), "%b-%d-%H-%M-%S")
            except ValueError:
                pass
        return datetime.min  # fallback to avoid crashing on bad format

    return sorted(txt_files, key=extract_datetime)
