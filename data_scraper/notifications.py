import logging
from datetime import datetime
from enum import Enum

import requests

from .utils import get_module_config

logger = logging.getLogger(__name__)

Status = Enum("Status", "Success Warning Error")

options = get_module_config("notifications")
try:
    webhook = options["slack_webhook"]
except KeyError as e:
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
