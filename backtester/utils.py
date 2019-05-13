import os


def get_data_dir():
    """Reads data path from environment variable $OPTIONS_DATA_PATH.
    If it is not set, defaults to `data/`
    """

    if "OPTIONS_DATA_PATH" in os.environ:
        data_dir = os.path.expanduser(os.environ["OPTIONS_DATA_PATH"])
    else:
        data_dir = "data"
        os.mkdir(data_dir)

    return data_dir
