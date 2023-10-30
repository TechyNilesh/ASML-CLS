from .helper import range_gen
from river import linear_model,naive_bayes,tree,neighbors,ensemble
from river import preprocessing
from river import feature_selection
from river import stats,optim

model_options_reg = [
    linear_model.LinearRegression(), 
    tree.HoeffdingAdaptiveTreeRegressor(max_depth=10),
    ensemble.AdaptiveRandomForestRegressor(n_models=3, seed=42),
    neighbors.KNNRegressor(window_size=100)
    ]

preprocessor_options_reg = [
    preprocessing.MinMaxScaler(),
    preprocessing.StandardScaler()]

feature_selection_options_reg = [
  feature_selection.PoissonInclusion(p=0.1,seed=42),
  feature_selection.VarianceThreshold(threshold=0),
  feature_selection.SelectKBest(similarity=stats.PearsonCorr())
  ]



hyperparameteers_options_reg = {
    'LinearRegression': {
        'l2': [0.0, 0.01, 0.001],
        'optimizer':[optim.Adam(),optim.SGD()]
    },
    'HoeffdingAdaptiveTreeRegressor': {
        'max_depth': range_gen(10,100,step=20),
        'grace_period': range_gen(10,500,step=50),
        'max_size':range_gen(5,10,step=2)
    },
    'AdaptiveRandomForestRegressor':{},
    'KNNRegressor':{
        'n_neighbors':range_gen(2,20,step=2),
        'window_size':range_gen(100,300,step=100),
        'aggregation_method':['mean','median','weighted_mean']
    },
    'StandardScaler': {
        'with_std':[True,False]
    },
    'MinMaxScaler': {},
    "SelectKBest":{
        "k": range_gen(2,10,step=2),
        "similarity":[stats.PearsonCorr()],
                  },
    "PoissonInclusion":{
        "p":range_gen(0.1,1.0,step=0.1,float_n=True)
    },
    "VarianceThreshold":
    {
        'threshold':range_gen(0.1,1.0,step=0.1,float_n=True),
        "min_samples":range_gen(2,10,step=1)
    }

}