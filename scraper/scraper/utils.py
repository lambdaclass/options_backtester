import hashlib
import os


def create_spider_path(spider_name):
    """Reads data path from environment variable `$SCRAPER_DATA_PATH`.
    If it is not set, defaults to `./data`, and under it creates a dir 
    `spider_name`.
    Returns the full path of that dir.
    """
    env_path = os.environ.get('SCRAPER_DATA_PATH', './data')
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


def file_hash_matches_data(file_path, data):
    """Returns True if `file_path` has the same MD5 hash as `data`"""
    file_hash = file_md5(file_path)
    data_md5 = hashlib.md5(data.encode()).hexdigest()
    return file_hash == data_md5


def file_md5(file, chunk_size=4096):
    md5 = hashlib.md5()
    with open(file, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)

    return md5.hexdigest()
