from sklearn.neural_network import MLPClassifier as MLPC

from .models import Model
from ..util import save_pickle, load_pickle


class FeedForwardNetwork(Model):

    def __init__(
            self,
            hidden_layer_sizes=(100,),
            activation='relu',  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            solver="adam",  # ‘lbfgs’, ‘sgd’, ‘adam’
            alpha=0.0001, # l2 penalty param
            batch_size="auto",
            learning_rate="constant", # ‘constant’, ‘invscaling’, ‘adaptive’
            learning_rate_init=0.001,
            power_t=0.5, # exponent for exp. lr decay
            max_iter=200, # max epochs
            shuffle=True, # shuffle batches
            tol=1e-4, # tolerance for loss/score decrease/increase
            momentum=0.9, # momentum for sgd
            nesterovs_momentum=True,
            early_stopping=False, # stops when n_iter_no_change epochs yielded no increase
            n_iter_no_change=10,
            validation_fraction=0.1, # percentage of training set used for validation
            beta_1=0.9, # beta1 for adam
            beta_2=0.999, # beta2 for adam
            epsilon=1e-8 # adam numerical stability 
        ):
        self._clf = MLPC(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            solver=solver, alpha=alpha, batch_size=batch_size,
            learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, shuffle=shuffle, tol=tol, early_stopping=early_stopping, n_iter_no_change=n_iter_no_change,
            momentum=momentum, nesterovs_momentum=nesterovs_momentum,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )
        self._last_score = None
        self._model_path = None

    def train(self, x, y, **kwargs):
        self._clf = self._clf.fit(x, y)
        return self.model

    def test(self, x, y, **kwargs):
        self._last_score = self._clf.score(x, y)
        return self._last_score

    def infer(self, x):
        return self._clf.predict(x)
        
    @property
    def model(self):
        return self._clf

    def __str__(self):
        return "A feed forward network with layers and width of hidden layers and other stuff."
