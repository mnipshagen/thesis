from sklearn.ensemble import RandomForestClassifier as RFC

from .models import Model
from ..util import save_pickle, load_pickle


class RandomForestModel(Model):

    def __init__(
            self,
            n_estimators=8000,
            max_features=None,
            bootstrap=True,
            class_weight="balanced",
            n_jobs=-1
        ):
        self._clf = RFC(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            n_jobs=n_jobs
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
        return f'RandomForest model working on {self.model.n_features} on {len(self.model.estimators_)} subtrees producing {self.model.n_outputs} outputs. The last score was {f"{self._last_score: .3f}" if self._last_score else "not performed yet"}.'
