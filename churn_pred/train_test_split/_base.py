from abc import ABC, abstractmethod


class BaseTrainTestSplit(ABC):
    @abstractmethod
    def split(self, df):
        pass
