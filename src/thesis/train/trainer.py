import json, os, time
from multiprocessing import Manager, Pool

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from ..model import ffn, random_forest as rf, decision_tree as dt, svm
from ..util import save_pickle, load_pickle


def read_configs(path="data/model_configs"):
    """
    Deprecated: Json files are no longer in use.

    Loads the json files, storing the parameter grid for the grid search
    """
    configs = os.listdir(path)
    print(f"Found {configs}")
    cfg_dict = {}
    for config_file in configs:
        print(f"Loading {config_file}...", end=" ")
        name = config_file.split(".")[0]
        with open(os.path.join(path, config_file), mode="r") as fh:
            data = json.load(fh)
        cfg_dict[name] = data
        print("Done.")
    
    return cfg_dict


def grid_params():
    """
    Convenience Funtion to return a dictionary of the grid search parameters
    """
    svm_params = [
        {
            "kernel" : ("linear", ),
            "C" : [1,10,100]
        },
        {
            "kernel" : ("rbf", "sigmoid"),
            "C" : [1, 10, 100],
            "gamma" : ["scale", 0.001, 0.0001, 0.00001],
            "coef0" : [0, 1]
        },
        {
            "kernel": ("poly", ),
            "C": [1, 10, 100],
            "degree": [3, 5, 7],
            "gamma": ["scale", 0.001, 0.0001, 0.00001],
            "coef0": [0, 1]
        }
    ]
    dtc_params = [
        {
            "min_samples_split" : [2, 3, 5],
            "min_samples_leaf" : [1, 2, 5],
        }
    ]
    rfc_params = [
        {
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 2, 5],
        }
    ]
    ffn_params = [
        {# n_iter_no_change=10, nsterov=True
            "hidden_layer_sizes" : [(256,1024,2048,512,16), (2048,512,128,16,8), (16,256,512,128,8), (512, 64, 8), (100)],
            "activation" : ["relu", "tanh", "logistic"],
            "solver" : ["adam"],
            "alpha" : [0.0001, 0.0005, 0.00001, 0.001],
            "batch_size" : [64, 256, 512],
            "learning_rate" : ["constant", "invscaling", "adaptive"],
            "learning_rate_init" : [0.001, 0.005, 0.01, 0.0005],
            "power_t" : [0.5, 0.8],
            "max_iter" : [25, 50, 100, 200, 500],
            "beta_1" : [0.85, 0.9, 0.95],
            "beta_2" : [0.9, 0.95, 0.99, 0.999], 
        },
        {
            "hidden_layer_sizes": [(256, 1024, 2048, 512, 16), (2048, 512, 128, 16, 8), (16, 256, 512, 128, 8), (512, 64, 8), (100)],
            "activation": ["relu", "tanh", "logistic"],
            "solver" : ["sgd"],
            "momentum": [0.7, 0.9, 0.99],
            "alpha": [0.0001, 0.0005, 0.00001, 0.001],
            "batch_size": [64, 256, 512],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": [0.001, 0.005, 0.01, 0.0005],
            "power_t": [0.5, 0.8],
            "max_iter": [25, 50, 100, 200, 500]
        }
    ]

    return {"svm":svm_params, "dtc":dtc_params, "rfc":rfc_params, "ffn":ffn_params}

def train(x, Y, test_size=0.2, subset=None, save_path="data/models"):
    """
    Train all models mentioned in subset on input x for targets y with the given test_size fraction.

    Saves models in the supplied folder. Does not return the whole grid search to save on memory. GridSearch is performed in parallel.

    Known models: svm, dtc, rfc

    x is expected to be a 2d numpy array or pandas dataframe
    y is expected to be a 1d numpy array or pandas series

    Note: The FFN training was sourced out to a iPython notebook.
    """
    print(f"Start Training!")
    models = {
        # "ffn": ffn.FeedForwardNetwork(),
        "rfc" : rf.RandomForestModel(class_weight="balanced", n_jobs=14),
        "dtc" : dt.DecisionTreeModel(class_weight="balanced"),
        "svm": svm.SupportVectorMachineModel(cache_size=1000, class_weight="balanced")
    }
    if subset is not None:
        subset = {k:v for k, v in models.items() if k in subset}
    else:
        subset = models
    print(f"Training: {list(subset.keys())}")

    scorers = {
        "accuracy" : make_scorer(balanced_accuracy_score),
        "precision" : make_scorer(precision_score, pos_label=2, average="binary"),
        "recall" : make_scorer(recall_score, pos_label=2, average="binary"),
        "f1" : make_scorer(f1_score, pos_label=2, average="binary" )
    }

    ohec = OneHotEncoder(categories="auto")
    x_ohec = ohec.fit_transform(x)
    train_data = {"dtc" : x, "rfc" : x, "ffn" : x_ohec, "svm" : x_ohec}

    search_params = grid_params()
    for model_name in subset.keys():
        print(f"GridSearching {model_name}...", flush=True)
        model = subset[model_name].model
        grid = GridSearchCV(
            estimator=model,
            param_grid=search_params[model_name],
            scoring=scorers,
            refit="recall",
            n_jobs=14,
            cv=5,
            verbose=3
        )
        t_start = time.time()
        result = grid.fit(train_data[model_name], Y)
        t_end = time.time()
        m, s = divmod((t_end - t_start), 60)
        h, m = divmod(m, 60)

        print(f"Done. Took {h}:{m:0>2}:{s:0>2}.", flush=True, end=" ")
        save_pickle(result, f"data/results/{model_name}_gridsearch")
        print(f"Saved.", flush=True)