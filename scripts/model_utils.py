import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, KFold
from typing import Union, Dict, Any, List, Tuple
import optuna as opt
import time

def objective_function(
    trial: opt.trial.Trial,
    *,
    model: BaseEstimator,
    params: Dict[str, Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, KFold] = 5,
    scoring: str = 'neg_root_mean_squared_error',
    negative: bool = True) -> float:
    """
    Objective function for Optuna study. Performs hyperparameter tuning
    using cross-validation for the provided model.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial object.
        model (BaseEstimator): The sklearn-compatible machine learning model to optimize.
        params (dict): A dictionary of hyperparameter search spaces.
            Example format:
            {
                'param_name': {'type': 'float', 'low': 0.01, 'high': 1.0, 'log': True},
                'param_name': {'type': 'int', 'low': 1, 'high': 10, 'step': 1},
                'param_name': {'type': 'categorical', 'choices': ['a', 'b', 'c']}
            }
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples, ).
        cv (Union[int, sklearn.model_selection.KFold], optional): Cross-validation strategy.
            Either the number of folds or a KFold object. Defaults to 5.
        scoring (str, optional): Scoring metric for cross-validation. Defaults to 'neg_root_mean_squared_error'.
        negative (bool, optional): Whether to negate the score (useful for metrics where higher is better). Defaults to True.

    Returns:
        float: The mean cross-validation score (adjusted for sign if `negative` is True).

    Example:
        >>> study.optimize(lambda trial: objective(
                trial,
                model=RandomForestRegressor(),
                params=param_space,
                X=X_train,
                y=y_train,
                cv=KFold(n_splits=5)
            ), n_trials=100)
    """
    model_clone = clone(model)

    suggested_params = {}
    for k, v in params.items():
        if v['type'] == 'float':
            suggested_params[k] = trial.suggest_float(k, v['low'], v['high'], log=v['log'])
        elif v['type'] == 'int':
            suggested_params[k] = trial.suggest_int(k, v['low'], v['high'], step=v['step'])
        elif v['type'] == 'categorical':
            suggested_params[k] = trial.suggest_categorical(k, v['choices'])

    if 'random_state' in model.get_params():
        suggested_params['random_state'] = 42

    model_clone.set_params(**suggested_params)

    scores = cross_val_score(
        estimator=model_clone,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return scores.mean() * (-1 if negative else 1)


def search_best_hyperparameters(
    models: Dict[str, object],
    params_grid: Dict[str, dict],
    trials: Dict[str, int],
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, KFold] = 5,
    metric: str = 'RMSE'
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """
    Searches for the best hyperparameters for each provided model using Optuna.

    For each model, the function runs an independent Optuna study to find the
    optimal set of hyperparameters based on cross-validation performance.

    Args:
        models (Dict[str, object]):
            A dictionary where keys are model names and values are sklearn-compatible model objects.
        params_grid (Dict[str, dict]):
            A dictionary where keys are model names and values are parameter search spaces for each model.
        trials (Dict[str, int]):
            A dictionary where keys are model names and values are the number of optimization trials for each model.
        X (np.ndarray):
            Feature matrix of shape (n_samples, n_features).
        y (np.ndarray):
            Target vector of shape (n_samples,).
        cv (Union[int, KFold], optional):
            Number of cross-validation folds or a cross-validation generator. Defaults to 5.
        metric (str, optional):
            Evaluation metric to optimize. Supported: 'RMSE', 'MSE'. Defaults to 'RMSE'.

    Returns:
        Tuple[Dict[str, dict], Dict[str, float]]:
            - Dictionary where keys are model names and values are the best-found hyperparameters for each model.
            - Dictionary where keys are model names and values are the best cross-validation scores (according to the selected metric) achieved during optimization.

    Raises:
        ValueError: If an unsupported evaluation metric is provided.

    Example:
        >>> best_params, best_scores = search_best_hyperparameters(
                models=model_dict,
                params_grid=param_spaces,
                trials={'XGBoost': 50, 'LGBM': 50},
                X=X_train,
                y=y_train,
                cv=KFold(n_splits=5),
                metric='RMSE'
            )
    """
    best_hyperparams = {}
    best_results = {}

    for model_name, model in models.items():

        metrics = {
            'RMSE': {'direction': 'minimize', 'scoring': 'neg_root_mean_squared_error', 'negative':True},
            'MSE': {'direction': 'minimize', 'scoring': 'neg_mean_squared_error', 'negative':True},
        }

        study = opt.create_study(
            direction=metrics[metric]['direction'],
            study_name=model_name,
        )

        study.optimize(
            lambda trial: objective_function(
                trial=trial,
                model=model,
                X=X,
                y=y,
                cv=cv,
                params=params_grid[model_name],
                scoring=metrics[metric]['scoring'],
                negative=metrics[metric]['negative'],
            ),
            n_trials=trials[model_name],
        )

        best_hyperparams[model_name] = study.best_params
        best_results[model_name] = study.best_value

        print('Model name:', model_name)
        print('Best parameters:', study.best_params)
        print(f'Best {metric}:', round(study.best_value, 6), end='\n\n')

        time.sleep(1)

    return best_hyperparams, best_results



def prepare_models(
    models: Dict[str, BaseEstimator],
    hyperparams: Dict[str, dict]
) -> List[Tuple[str, BaseEstimator]]:
    """
    Clone and configure machine learning models with their specified hyperparameters.

    This function takes a dictionary of model names and their corresponding model instances,
    clones each model to avoid mutating the original objects, sets the provided hyperparameters,
    and returns a list of (model_name, model_instance) tuples.

    Args:
        models (Dict[str, RegressorMixin]):
            Dictionary where keys are model names and values are untrained sklearn-compatible models.
        hyperparams (Dict[str, dict]):
            Dictionary where keys are model names and values are dictionaries of hyperparameters
            to apply to each corresponding model.

    Returns:
        List[Tuple[str, RegressorMixin]]:
            List of tuples, each containing a model name and the corresponding model instance with
            the specified hyperparameters.

    Raises:
        KeyError: If the provided hyperparameters do not match the model names in the `models` dictionary.

    Example:
        >>> models = {'LinearRegression': LinearRegression(), 'Ridge': Ridge()}
        >>> hyperparams = {'LinearRegression': {'fit_intercept': True}, 'Ridge': {'alpha': 1.0}}
        >>> prepared_models = prepare_models(models, hyperparams)
    """
    base_models = []

    for model_name, model in models.items():
        if model_name not in hyperparams:
            raise KeyError(f"Hyperparameters for model '{model_name}' are not provided.")

        model_clone = clone(model)
        model_clone.set_params(**hyperparams[model_name])
        base_models.append((model_name, model_clone))

    return base_models
