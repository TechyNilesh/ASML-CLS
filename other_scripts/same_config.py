from river import linear_model,naive_bayes,tree,neighbors,preprocessing

model_options = [
    linear_model.LogisticRegression(),
    naive_bayes.GaussianNB(),
    tree.HoeffdingTreeClassifier(),
    neighbors.KNNClassifier()]

preprocessor_options = [preprocessing.MinMaxScaler(),
                            preprocessing.StandardScaler(),
                            preprocessing.MaxAbsScaler()]

feature_selection_options = []

hyperparameters_options = {
    "LogisticRegression": {
        "l2": [.0,.01,.001],
    },
    "GaussianNB": {},
    "HoeffdingTreeClassifier": {
        "max_depth": [10, 30, 60, 10, 30, 60],
        "grace_period": [10, 100, 200, 10, 100, 200],
        "max_size": [5, 10],
    },
    "KNNClassifier": {
        "n_neighbors": [1, 5, 20],
        "window_size": [100, 500, 1000],
        "weighted": [True, False],
    },
    'MinMaxScaler':{},
    'StandardScaler':{},
    'MaxAbsScaler':{},
}

config_dict = {}

config_dict['models'] = model_options
config_dict['preprocessors'] = preprocessor_options
config_dict['features'] = feature_selection_options 
config_dict['hyperparameters'] = hyperparameters_options