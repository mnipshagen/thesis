from sklearn.tree import DecisionTreeClassifier as DTC

from .models import Model
from ..util import save_pickle, load_pickle


class DecisionTreeModel(Model):

    def __init__(
            self,
            class_weight="balanced"
    ):
        self._clf = DTC(
            class_weight=class_weight
        )
        self._last_score = None
        self._model_path = None

    def train(self, x, y, **kwargs):
        self._clf = self._clf.fit(x, y)
        return self.model

    def test(self, x, y, **kwargs):
        self._last_score = self._clf.score(x, y)
        return self._last_score

    def infer(self, x, **kwargs):
        self._clf.predict(x)

    @property
    def model(self):
        return self._clf

    def __str__(self):
        return f"DecisionTree model working on {self.model.n_features} features producing {self.model.n_outputs} outputs. The last score was {f'{self._last_score:.3f}' if self._last_score else 'not performed yet'}."
