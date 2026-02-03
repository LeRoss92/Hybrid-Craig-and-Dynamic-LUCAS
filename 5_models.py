from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, QuantileRegressor, ElasticNet, BayesianRidge, Ridge, TweedieRegressor, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_models(seed=4210):
    """
    Initialize and return the MODELS dictionary.
    
    Parameters
    ----------
    seed : int, default=4210
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary of model configurations.
    """
    MODELS = {
        # ============================== Fast ==============================
        'LinReg': {
            'model': LinearRegression(),
            'params': {},
        },
        'Piecewise_Linear_Reg': {
            'model': Pipeline([
                ("spline", SplineTransformer(degree=1,n_knots=3,include_bias=False)),
                ("interactions", PolynomialFeatures(degree=2, interaction_only=True, # no squared terms (x1 with x1)
                    include_bias=False)), # already exists
                ("ridge", Ridge(alpha=1.0)) ]),
            'params': {
                'spline__n_knots': [2, 3, 4],                   # 2 = linear regression, 3=two pieces
                'spline__degree': [1],                       # linear vs polynomial splines
                'spline__include_bias': [False],                # True: x=y=0
                'interactions__degree': [1, 2],                 # 1: no interactions, 2: pairwise...
                'ridge__alpha': [1e-6, 0.01, 0.1],    # 1e-6: minimal regularization for numerical stability
            },
            'n_jobs': 32,  # Use workers for GridSearchCV (when not parallelizing candidates)
            'n_jobs_feature_selection': 32  # Parallelize candidate feature evaluation (one candidate per worker)
        },
        'DecisionTree': {
            'model': DecisionTreeRegressor(random_state=seed),
            'params': {'max_depth': [5, 7, 9], 'min_samples_split': [10, 20], 'min_samples_leaf': [5, 10]},
            'n_jobs': 32,  # Use workers for GridSearchCV (when not parallelizing candidates)
            'n_jobs_feature_selection': 32  # Parallelize candidate feature evaluation (one candidate per worker)
        }, 































        'PLSR': {
            'model': PLSRegression(n_components=1),
            'scaler': StandardScaler(),
            'params': {'n_components': [1, 2]},
            'n_iter': 2
        },
        'ElasticNet': {
            'model': ElasticNet(max_iter=2000),
            'scaler': StandardScaler(),
            'params': {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.3, 0.5, 0.7]},
            'n_iter': 8
        },
        'BayesianRidge': { # = Ridge + uncertainty = GAM_Linear ...says ChatGPT
            'model': BayesianRidge(max_iter=300, compute_score=True),
            'scaler': StandardScaler(),
            'params': {'alpha_1': [1e-6, 1e-5, 1e-4], 'alpha_2': [1e-6, 1e-5, 1e-4], 'lambda_1': [1e-6, 1e-5, 1e-4], 'lambda_2': [1e-6, 1e-5, 1e-4]},
            'n_iter': 6
        },
        'SVR': { # although very slow for large sample sizes
            'model': SVR(),
            'scaler': StandardScaler(),
            'params': {'C': [0.1, 1.0, 10.0, 100.0], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['rbf', 'linear', 'poly']},
            'n_iter': 6
        }, 
        

        # ============================== Slow ==============================
        'LightGBM': {
            'model': LGBMRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, num_leaves=31, subsample=0.8,
                                   colsample_bytree=0.8, random_state=seed, verbosity=-1, device='gpu', force_col_wise=True, n_jobs=1),
            'scaler': None,
            'params': {},
            'n_iter': 1
        },
        'QuantileReg': {
            'model': QuantileRegressor(quantile=0.5, alpha=0.01, solver='revised simplex'),
            'scaler': None,
            'params': {'alpha': [0.0, 0.01, 0.05]},
            'n_iter': 3
        },
        'RF': {
            'model': RandomForestRegressor(random_state=seed, n_jobs=-1),
            'scaler': None,
            'params': {'n_estimators': [50, 100], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
            'n_iter': 10
        },
        'NeuralNet': {
            'model': MLPRegressor(random_state=seed, max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20),
            'scaler': StandardScaler(),
            'params': {'hidden_layer_sizes': [(32,), (64,), (32, 16)], 'activation': ['relu'], 'alpha': [0.001, 0.01], 'learning_rate_init': [0.001, 0.01]},
            'n_iter': 6
        },
        'XGB': {
            'model': XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                  tree_method='gpu_hist', predictor='gpu_predictor', n_jobs=-1, verbosity=0, random_state=seed), # 
            'scaler': None,
            'params': {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [6, 8], 'subsample': [0.8], 'colsample_bytree': [0.8]},
            'n_iter': 4
        },
        


        # ============================== Redundant ==============================
        'CatBoost': {
            'model': CatBoostRegressor(iterations=50, learning_rate=0.1, depth=6, random_seed=seed, verbose=False),
            'scaler': None,
            'params': {
                'iterations': [50, 100],
                'learning_rate': [0.05, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            },
            'n_iter': 6
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'scaler': StandardScaler(),
            'params': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'p': [1, 2]},
            'n_iter': 6
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=seed),
            'scaler': None,
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]},
            'n_iter': 8
        },
        'GAM_Tweedie': {
            'model': Pipeline([
                ("splines", SplineTransformer(n_knots=10, degree=3, include_bias=False)),
                ("tweedie", TweedieRegressor(power=1.5, alpha=1.0, max_iter=200))
            ]),
            'scaler': StandardScaler(),
            'params': {
                'splines__n_knots': [5, 10, 20],
                'tweedie__power': [1.0, 1.5, 2.0],  # 1=Poisson-like, 1.5=Tweedie, 2=Gamma-like
                'tweedie__alpha': [0.1, 1.0, 10.0]
            },
            'n_iter': 6
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=seed, n_jobs=-1),
            'scaler': None,
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2', None]
            },
            'n_iter': 8
        },
        'GAM_ShapeConstrained': {
            'model': Pipeline([
                ("splines", SplineTransformer(n_knots=10, degree=3, include_bias=False)),
                ("ridge", Ridge(alpha=1.0))
            ]),
            'scaler': StandardScaler(),
            'params': {
                'splines__n_knots': [5, 10, 20],
                'ridge__alpha': [1.0, 10.0, 100.0]  # Higher regularization for smoother, more constrained fits
            },
            'n_iter': 6
        },
        'GAM_Spline': {
            'model': Pipeline([
                ("splines", SplineTransformer(n_knots=10, degree=3, include_bias=False)),
                ("ridge", Ridge(alpha=1.0))
            ]),
            'scaler': StandardScaler(),
            'params': {
                'splines__n_knots': [5, 8],
                'ridge__alpha': [1.0, 10.0]
            },
            'n_iter': 6
        },
        'GAM_EBM': {
            'model': GradientBoostingRegressor(max_depth=2, learning_rate=0.01, n_estimators=100, random_state=seed),
            'params': {
                'learning_rate': [0.01, 0.05],
                'n_estimators': [100, 300],
                'subsample': [1.0]
            }
        },
    }
    return MODELS

