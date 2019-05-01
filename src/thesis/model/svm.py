from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder as OHE

from .models import Model
from ..util import save_pickle, load_pickle


class SupportVectorMachineModel(Model):

    def __init__(
        self,
        C=1.0,  # Penalty parameter C of the error term.
        kernel="linear", # must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
        degree=3,  # Degree of the polynomial kernel function (‘poly’)
        gamma="scale",  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        coef0=0.0, # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        cache_size=200,  # Specify the size of the kernel cache (in MB).
        class_weight="balanced" 
    ):
        self._clf = SVC(
            C=C, kernel=kernel, degree=degree,
            gamma=gamma, coef0=coef0, cache_size=cache_size,
            class_weight=class_weight
        )
        self._encoder = None

    def train(self, x, y, **kwargs):
        # if any((x > 1) | (x < 1)):
        #     x = self.encode(x)
        self._clf = self._clf.fit(x, y)
        return self.model
    
    def test(self, x, y, **kwargs):
        self._last_score = self._clf.score(x, y)
        return self._last_score

    def infer(self, x, **kwargs):
        self._clf.predict(x)

    def encode(self, x):
        if not self._encoder:
            self._encoder = OHE(sparse=False, categories="auto")
        return self._encoder.fit_transform(x)

    @property
    def model(self):
        return self._clf

    def __str__(self):
        return f'SupportVectorMachine model.'
