from abc import ABCMeta, abstractmethod


class Strategy(metaclass=ABCMeta):
    """Interface for the different investing strategies"""

    @abstractmethod
    def generate_signals(self, event):
        """Provides the mechanisms to calculate the list of signals.
        """
        raise NotImplementedError("Strategy must implement generate_signals()")
