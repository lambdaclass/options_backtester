import math
from functools import reduce

import pandas as pd
import numpy as np

from .direction import Direction
from .asset import Asset


class Strategy:
    def __init__(self, direction=Direction.BUY):
        assert isinstance(direction, Direction)
        self.direction = direction
        self.assets = []

    def add_asset(self, asset):
        """Adds asset to the strategy"""
        assert isinstance(asset, Asset)
        self.assets.append(asset)
        return self

    def add_assets(self, assets):
        """Adds assets to the strategy"""
        for asset in assets:
            self.add_asset(asset)
        return self

    def remove_asset(self, asset_number):
        """Removes asset from the strategy"""
        self.assets.pop(asset_number)
        return self

    def clear_assets(self):
        """Removes *all* assets from the strategy"""
        self.assets = []
        return self
