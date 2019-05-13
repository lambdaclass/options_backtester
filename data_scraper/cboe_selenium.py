# CBOE data scraper
# Requires Selenium and a headless Chrome driver

import tempfile
import time
import os
import shutil
from datetime import date
from selenium import webdriver


class CBOE():
    """CBOE data downloader."""
    url = "http://www.cboe.com/delayedquote/quote-table-download"

    def __init__(self):
        self.data_path = self._get_data_path()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.driver = self._initilize_driver(self.tmp_dir.name)

    def _get_data_path():
        path = os.getenv("OPTIONS_DATA_PATH")
        if not path:
            raise EnvironmentError("Environment variable $OPTIONS_DATA_PATH not set")
        return os.path.expanduser(path)

    def _initilize_driver(download_dir):
        """Initilizes the Chrome driver to silently download files
        to a temporary directory.
        """
        options = webdriver.ChromeOptions()
        options.add_argument("headless")
        options.add_argument("disable-gpu")

        driver = webdriver.Chrome(options=options)
        driver.command_executor._commands["send_command"] = (
            "POST",
            "/session/$sessionId/chromium/send_command"
        )
        params = {
            "cmd": "Page.setDownloadBehavior",
            "params": {
                "behavior": "allow",
                "downloadPath": download_dir
            }
        }
        driver.execute("send_command", params)
        driver.implicitly_wait(10)
        return driver

    def fetch_data(self, symbols):
        """Fetches options data for a given list of symbols"""
        self.driver.get(CBOE.url)
        for symbol in symbols:
            ticker = self.driver.find_element_by_css_selector("input#txtTicker")
            ticker.send_keys(symbol)
            submit = self.driver.find_element_by_css_selector("input#cmdSubmit")
            submit.click()
            time.sleep(15)  # Horrible hack
            download_path = os.path.join(self.tmp_dir.name, "quotedata.dat")
            renamed_file = date.today().strftime(symbol + "_%Y%m%d.csv")
            full_path = os.path.join(self.data_path, renamed_file)
            shutil.move(download_path, full_path)

    def __del__(self):
        self.tmp_dir.cleanup()
