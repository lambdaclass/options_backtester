import os


def create_spider_path(spider_name):
    """Reads data path from environment variable `$SAVE_DATA_PATH`.
    If it is not set defaults to `./data`, and under it creates a dir 
    `spider_name`.
    Returns the full path of that dir.
    """
    env_path = os.environ.get('SAVE_DATA_PATH', './data')
    save_data_path = os.path.realpath(os.path.expanduser(env_path))
    spider_path = os.path.join(save_data_path, spider_name)

    if not os.path.exists(spider_path):
        os.makedirs(spider_path)

    return spider_path


def get_log_path(filename='scraper.log'):
    """Returns absolute path of the `$LOG_FILE`"""
    log_dir = os.environ.get('SCRAPER_LOG_PATH', '/var/log/scraper')
    log_dir = os.path.realpath(os.path.expanduser(log_dir))

    return os.path.join(log_dir, filename)
