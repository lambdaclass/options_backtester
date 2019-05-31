import logging
from datetime import datetime
from enum import Enum

import requests

from .utils import get_module_config, get_environment_var

logger = logging.getLogger(__name__)

Status = Enum("Status", "Success Warning Error")

options = get_module_config("notifications")
try:
    webhook = get_environment_var("SLACK_WEBHOOK")
except EnvironmentError as e:
    logger.error("Missing slack webhook from configuration file")
    raise e

payload = {
    "channel": "#algotrading",
    "username": "Talebot",
    "icon_emoji": ":taleb:",
    "attachments": [{
        "footer": "Talebot"
    }]
}


def slack_notification(text, scraper, status=Status.Error):
    """Post Slack notification"""

    if status == Status.Error:
        emoji = ":thumbsdown: "
        title = "data_scraper error"
        color = "#B22222"
    else:
        title = "data_scraper status report"
        if status == Status.Success:
            emoji = ":thumbsup: "
            color = "#49C39E"
        else:
            emoji = ":warning: "
            color = "#EDB625"
    msg = emoji + text

    payload["attachments"][0]["fallback"] = msg
    payload["attachments"][0]["text"] = msg
    payload["attachments"][0]["color"] = color
    payload["attachments"][0]["title"] = title
    payload["attachments"][0]["fields"] = [{"title": scraper}]
    payload["attachments"][0]["ts"] = datetime.today().timestamp()

    response = requests.post(webhook, json=payload)

    if response.status_code != 200:
        msg = "Error connecting to Slack {}. Response is:\n{}".format(
            response.status_code, response.text)
        logger.error(msg)


def send_report(done, failed, scraper, op="scrape"):
    """Sends status report to Slack for given operation.
    `done` is the count of successfully scraped/aggregated symbols
    `failed` is a list of symbol names that could not be scraped/aggregated
    """
    if done > 0:
        msg = "Successfully {}d {}".format(op, _symbol_str(done))
        slack_notification(msg, scraper, status=Status.Success)
    if len(failed) > 0:
        msg = "Failed to {} {}: {}".format(op, _symbol_str(len(failed)),
                                           ", ".join(failed))
        slack_notification(msg, scraper, status=Status.Warning)


def _symbol_str(count):
    return str(count) + " symbol" if count == 1 else str(count) + " symbols"
