from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    @abstractmethod
    def fit(self, model, objective):
        pass

    @abstractmethod
    def explain_decision_plot(self, X_tab_explain):
        pass
