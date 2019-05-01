from abc import ABC, abstractmethod

# from . import ffn.FullyConnectedModel as FCM, decision_tree.DecisionTreeModel as DTM, random_forest.RandomForestModel as RFM, svm.SupportVectorModel as SVM
from ..util import save_pickle, load_pickle

# MODELS = [FCM, DTM, RFM, SVM]

class Model(ABC):

    def __init__(self):
        self._last_score = None
        self._model_path = None

    @abstractmethod
    def train(self, X, y, **kwargs):
        pass

    @abstractmethod
    def test(self, X, y, **kwargs):
        pass

    @abstractmethod
    def infer(self, x, **kwargs):
        pass
    
    @property
    @abstractmethod
    def model(self):
        pass

    def save(self, path, save_path=True):
        save_pickle(self.model, path)
        if save_path:
            self._model_path = path

    def load(self, path):
        if path:
            self._clf = load_pickle(path)
        elif self._model_path:
            self._clf = load_pickle(self._model_path)
        else:
            raise ValueError("No path supplied.")
        return True

    def __str__(self):
        return "I am a machine learning model"
