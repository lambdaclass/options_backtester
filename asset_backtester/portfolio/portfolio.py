from .direction import Direction
from .asset import Asset


class Portfolio:
    def __init__(self, direction=Direction.BUY):
        assert isinstance(direction, Direction)
        self.direction = direction
        self.assets = []

    def add_asset(self, asset):
        """Adds asset to the Portfolio"""
        assert isinstance(asset, Asset)
        self.assets.append(asset)
        return self

    def add_assets(self, assets):
        """Adds assets to the Portfolio"""
        for asset in assets:
            self.add_asset(asset)
        return self

    def remove_asset(self, asset_number):
        """Removes asset from the Portfolio"""
        self.assets.pop(asset_number)
        return self

    def clear_assets(self):
        """Removes *all* assets from the Portfolio"""
        self.assets = []
        return self
