import random
import numpy as np
from river import metrics
from river import utils
from river import base
from .models import ProgressiveModelSelector, DynamicEnsembleClassifier, DynamicEnsembleRegressor
from .clf_config import model_options, preprocessor_options, feature_selection_options, hyperparameters_options
from .reg_config import model_options_reg, preprocessor_options_reg, feature_selection_options_reg, hyperparameteers_options_reg

import logging


class AutoStreamClassifier(base.Estimator):
    def __init__(
        self,
        config_dict=None,
        metric=metrics.Accuracy(),
        prediction_mode="ensemble",
        ensemble_size=3,
        exploration_window=1000,
        budget=10,
        update_hist=False,
        return_metric=True,
        verbose=False,
        mode="greedy",
        no_fs_models=[],
        seed=42):
        """
        AutoStreamML class for Searching best online/streaming machine learning pipeline Combination.

        Parameters:
        - metric (object, optional): Evaluation metric for model performance (default: Accuracy).
        - prediction_mode (str, optional): 'ensemble' or 'best' (default: 'ensemble').
        - ensemble_size (int, optional): Maximum size of the dynamic ensemble (default: 3).
        - exploration_window (int, optional): Size of the exploration window (default: 1000).
        - budget (int, optional): Total budget for pipeline operations (default: 10).
        - update_hist (bool, optional): Whether to update the performance history (default: False).
        - return_metric (bool, optional): weather to return the performence metrics or not (default: True).
        - verbose (bool, optional): Whether to print verbose output (default: False).
        - mode (str, optional): 'greedy' or 'fast' (default: greedy)
        - no_fs_models (list,optional): Skip feature selection for listed models due to error or Not Needed (default: []).
        - seed (int, optional): Random seed for reproducibility (default: 42).
        """
        # Validate metric
        if not isinstance(metric, metrics.base.Metric):
            raise ValueError(f"The 'metric' parameter must be a valid metric object from sklearn.metrics. Received: {metric}, {type(metric)}")

        # Validate prediction_mode
        if prediction_mode not in ["ensemble", "best"]:
            raise ValueError(f"The 'prediction_mode' parameter must be 'ensemble' or 'best'. Received: {prediction_mode}, {type(prediction_mode)}")

        # Validate ensemble_size
        if not isinstance(ensemble_size, int) or ensemble_size < 1:
            raise ValueError(f"The 'ensemble_size' parameter must be a positive integer. Received: {ensemble_size}, {type(ensemble_size)}")

        # Validate exploration_window
        if not isinstance(exploration_window, int) or exploration_window < 1:
            raise ValueError(f"The 'exploration_window' parameter must be a positive integer. Received: {exploration_window}, {type(exploration_window)}")

        # Validate budget
        if not isinstance(budget, int) or budget < 1:
            raise ValueError(f"The 'budget' parameter must be a positive integer. Received: {budget}, {type(budget)}")

        # Validate update_hist
        if not isinstance(update_hist, bool):
            raise ValueError(f"The 'update_hist' parameter must be a boolean (True or False). Received: {update_hist}, {type(update_hist)}")

        # Validate return_metric
        if not isinstance(return_metric, bool):
            raise ValueError(f"The 'return_metric' parameter must be a boolean (True or False). Received: {return_metric}, {type(return_metric)}")

        # Validate verbose
        if not isinstance(verbose, bool):
            raise ValueError(f"The 'verbose' parameter must be a boolean (True or False). Received: {verbose}, {type(verbose)}")

        # Validate mode
        if mode not in ["greedy", "fast"]:
            raise ValueError(f"The 'mode' parameter must be 'greedy' or 'fast'. Received: {mode}, {type(mode)}")

        # Validate no_fs_models
        if not isinstance(no_fs_models, list):
            raise ValueError(f"The 'no_fs_models' parameter must be list of models names: {no_fs_models}, {type(no_fs_models)}")

        # Validate seed
        if not isinstance(seed, int):
            raise ValueError(f"The 'seed' parameter must be an integer. Received: {seed}, {type(seed)}")

        #Logger Init
        log_filename = "AutoStreamML.log"
        logging.basicConfig(filename=log_filename,level=logging.ERROR,format="[%(levelname)s] - %(message)s",force=True,filemode='w')

        self.metric = metric
        self.exploration_window = exploration_window
        self.verbose = verbose
        self.seed = seed
        self.COUNTER = 0
        self.EW_COUNTER = 0
        self.pipeline_list = []
        self.model = None
        self.ensemble_size = ensemble_size
        self.return_metric = return_metric
        self.mode = mode

        self.no_fs_models = no_fs_models

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.prediction_mode = prediction_mode

        self.update_hist = update_hist

        self.model_performance_dict = {}

        self.budget = budget

        # Allocate budget for the best pipeline (fixed value)
        self.best_pipeline_budegt = 1

        # Calculate the remaining budget after allocating to the best pipeline
        remaining_budget = self.budget - self.best_pipeline_budegt

        self.new_pipeline_budget = round(0.4 * remaining_budget)
        self.hyper_mutation_budegt = round(0.6 * remaining_budget)

        self.algorithms = []
        self.preprocessing_steps = []
        self.feature_selection_methods = []

        if config_dict==None:
            self.hyperparameters = hyperparameters_options
            self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in model_options]
            self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in preprocessor_options]
            self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in feature_selection_options]
        else:
            self.hyperparameters = config_dict['hyperparameters']
            self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in config_dict['models']]
            self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in config_dict['preprocessors']]
            self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in config_dict['fetures']]

        self.ensemble_model = DynamicEnsembleClassifier(models=[],
                                                        max_size=self.ensemble_size,
                                                        mode=self.mode,
                                                        metric=self.metric.clone())

        self._create_pipelines()

        self.model_selector = ProgressiveModelSelector(models=self.pipeline_list,
                                                       verbose=False,
                                                       mode=self.mode,
                                                       metric=self.metric.clone())

        self.ensemble_model.add_model(self.model_selector.best_model)

        self.predic_best = False

        if self.prediction_mode != "ensemble":
            self.predic_best = True
            self.best = self.model_selector.best_model

    def _get_hyperparameters(self, model, hyper_init="random"):
        """
        Get hyperparameters for a algorithm.

        Parameters:
        - algorithm: The selected algorithm.

        Returns:
        - hyperparameters_with_algo: Hyperparameters for the algorithm.
        """
        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]

        hyperparams = {}

        if hyper_init == "random":
            for key, values in model_hyper.items():
                hyperparams[key] = random.choice(values)
        else:
            default_hyper = model._get_params()
            for key, values in model_hyper.items():
                if default_hyper[key] in values:
                    hyperparams[key] = default_hyper[key]
                else:
                    hyperparams[key] = values[0]

        return model._set_params(hyperparams)

    def _create_pipelines(self):
        """
        Create a list of machine learning pipelines combining preprocessing, algorithms, and feature selection.
        """
        for preprocessing_step in self.preprocessing_steps:
            for model_instance in self.algorithms:
                # Create a pipeline without feature selection
                pipeline = preprocessing_step | model_instance
                self.pipeline_list.append(pipeline.clone())

                if type(model_instance).__name__ not in self.no_fs_models:
                    # Create a pipeline with feature selection
                    for feature_selection in self.feature_selection_methods:
                        full_pipeline = preprocessing_step | feature_selection | model_instance
                        self.pipeline_list.append(full_pipeline.clone())

    def _initialize_random_pipeline(self):
        """
        Initialize a random machine learning pipeline.

        Returns:
        - random_pipeline: A randomly initialized pipeline.
        """

        # Ensure that each component type is selected at least once
        random_algorithm = random.choice(self.algorithms)
        random_preprocessing = random.choice(self.preprocessing_steps)
        random_feature_selection = random.choice(self.feature_selection_methods)

        new_choice = random.choice(["WITH_FEATURE", "WITHOUT_FEATURE"])

        if type(random_algorithm).__name__ in self.no_fs_models:
            new_choice = "WITHOUT_FEATURE"

        if new_choice == "WITH_FEATURE":
            random_feature_selection = random.choice(self.feature_selection_methods)
            return random_preprocessing | random_feature_selection | random_algorithm
        else:
            return random_preprocessing | random_algorithm

    def _get_current_params(self, model):
        """
        Get the current hyperparameters of a model.

        Parameters:
        - model: The machine learning model.

        Returns:
        - current_hyper: Current hyperparameters of the model.
        """

        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]
        current_hyper = {}
        for k, _ in model_hyper.items():
            current_hyper[k] = model._get_params()[k]
        return current_hyper

    def _ardns(self, current_value, values_list):
        """
        Adaptive Random Directed Nearby Search (ARDNS) function.

        Parameters
        ----------
        current_value : any
            The current value that you want to find a nearby option for.

        values_list : list
            A list of possible values to choose from.

        Returns
        -------
        any
            The nearby value selected based on the random directed nearby search strategy.
        """

        nearby_option = np.random.choice(["same", "upper", "lower", "random"])

        if nearby_option == "same":
            return current_value
        elif nearby_option == "upper":
            return values_list[min(values_list.index(current_value) + 1, len(values_list) - 1)]
        elif nearby_option == "lower":
            return values_list[max(values_list.index(current_value) - 1, 0)]
        else:  # "random"
            return np.random.choice(values_list)

    def _suggest_nearby_hyperparameters(self, model):
        """
        Suggest nearby hyperparameter values by perturbing the current values cyclically within a defined range.

        Parameters:
        - model: The machine learning model instance.

        Returns:
        - model: A copy of the model with suggested hyperparameter values.
        """

        # Get the current hyperparameter values
        current_hyperparameters = self._get_current_params(model)

        # Get the user define hyperparameter values
        user_defined_search_space = self.hyperparameters[type(model).__name__]
        # print(user_defined_search_space)

        suggested_hyperparameters = {}

        for param_name, param_value in current_hyperparameters.items():
            if isinstance(param_value, int):
                # Integer hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = int(new_value)

            elif isinstance(param_value, float):
                # Float hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = float(new_value)

            elif isinstance(param_value, (str, bool)):
                # Categorical hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = new_value

            else:
                # Unsupported type (e.g., NoneType)
                # random value selected from search space
                suggested_hyperparameters[param_name] = random.choice(user_defined_search_space[param_name])

        return model._set_params(suggested_hyperparameters)
        # return model.clone(new_params=suggested_hyperparameters)

    def next_nerby(self, pipeline):
        """
        Mutate a pipeline by perturbing hyperparameters.

        Parameters:
        - pipeline: The original machine learning pipeline.

        Returns:
        - next_nerby_pipeline: The mutated machine learning pipeline.
        """

        next_nerby_pipeline = list(pipeline.steps.values())

        new_algorithm = self._suggest_nearby_hyperparameters(next_nerby_pipeline[-1])
        new_preprocessing = self._suggest_nearby_hyperparameters(next_nerby_pipeline[0])

        if len(next_nerby_pipeline) > 2:
            new_feature_selection = self._suggest_nearby_hyperparameters(next_nerby_pipeline[1])
            return new_preprocessing | new_feature_selection | new_algorithm
        else:
            return new_preprocessing | new_algorithm

    def select_and_update_pipelines(self):
        """
        Select and update pipelines based on performance and budgets.
        """

        best_pipeline = self.best_model.clone()

        next_nerby_pipelines = []

        for _ in range(self.hyper_mutation_budegt):
            next_nerby_pipeline = self.next_nerby(best_pipeline)
            next_nerby_pipelines.append(next_nerby_pipeline)
            best_pipeline = next_nerby_pipeline.clone()

        if self.update_hist:
            new_pipelines = [self.select_models_probabilistically().clone() for i in range(self.new_pipeline_budget)]
        else:
            new_pipelines = [self._initialize_random_pipeline().clone() for i in range(self.new_pipeline_budget)]

        self.pipeline_list = [self.best_model.clone()] + next_nerby_pipelines + new_pipelines

        # maintaining uniquness
        self.pipeline_list = list(set(self.pipeline_list))

    def select_models_probabilistically(self):
        """
        Randomly select models with probabilities based on their historical performance scores.

        Returns:
        - selected_model: The selected model.
        """

        # Extract models and their performance scores from the dictionary
        models = list(self.model_performance_dict.keys())
        performance_scores = list(self.model_performance_dict.values())

        # Calculate the total performance score
        total_performance_score = sum(performance_scores)

        # Calculate the selection probabilities based on performance scores
        selection_probabilities = [score / total_performance_score for score in performance_scores]

        # Randomly select a model based on the calculated probabilities
        selected_model_idx = random.choices(range(len(models)), weights=selection_probabilities)[0]
        selected_model = models[selected_model_idx]

        return selected_model

    def reset_exploration(self):
        """
        Reset the exploration mechanism.
        """
        self.model_selector.update_models(self.pipeline_list)

    def print_batch_info(self):
        """
        Print information about the current batch and model performance.
        """
        print(
            f"Data Point: {self.COUNTER} | Exploration Window: {self.EW_COUNTER}")
        try:
            print(f"Best Pipeline: {self.best_model}")
            print(f"Best Preprocessor Hyper: {self._get_current_params(list(self.best_model.steps.values())[0])}")
            if len(list(self.best_model.steps.values())) == 3:
                print(f"Best Feature Hyper: {self._get_current_params(list(self.best_model.steps.values())[1])}")
            print(f"Best Model Hyper: {self._get_current_params(list(self.best_model.steps.values())[-1])}")
        except Exception as e:
            logging.error(f"[{self.COUNTER}] - [Print Section] - {e}")
        if self.return_metric:
            print(f"Best Pipeline Score: {self.metric.get() * 100:0.2f}%")
        print("----------------------------------------------------------------------")

    @property
    def best_model(self):
        """
        Get the best-performing model.

        Returns:
        - best_model: The best-performing machine learning model.
        """
        return self.model_selector.best_model

    def predict_one(self, x):
        """
        Make a prediction for a single data point.

        Parameters:
        - x: Input data.

        Returns:
        - y_pred: Predicted output.
        """
        try:
            if self.predic_best:
                try:
                    return self.best.predict_one(x)
                except Exception as e:
                    logging.error(f"[{self.COUNTER}] - [Predic Best Section] - {e}")
                    return None
            return self.ensemble_model.predict_one(x)
        except:
            logging.error(f"[{self.COUNTER}] - [Predic Ensemble Section] - {e}")
            return None

    def _exploration(self, x, y):
        """
        Perform exploration on incoming data.

        Parameters:
        - x: Input data.
        - y: Target labels.
        """
        # Exploration Time
        self.model_selector.learn_one(x, y)

    def _exploitation(self, x, y):
        """
        Perform exploitation on incoming data.

        Parameters:
        - x: Input data.S
        - y: Target labels.
        """
        # Exploitation Time
        if self.predic_best:
            try:
                if self.return_metric:
                    y_pred = self.best.predict_one(x)
                    self.metric.update(y, y_pred)
                self.best.learn_one(x, y)
            except Exception as e:
                logging.error(f"[{self.COUNTER}] - [Explotiation Section] - {e}")
        else:
            try:
                if self.return_metric:
                    y_pred = self.ensemble_model.predict_one(x)
                    self.metric.update(y, y_pred)
                self.ensemble_model.learn_one(x, y)
            except Exception as e:
                logging.error(f"[{self.COUNTER}] - [Explotiation Section] - {e}")

    def learn_one(self, x, y):
        """
        Learn from a single data point.

        Parameters:
        - x: Input data.
        - y: Target labels.
        """

        if self.COUNTER % self.exploration_window == 0:
            if self.predic_best:
                self.best = self.best_model

            if self.verbose:
                self.print_batch_info()

            if self.COUNTER != 0:
                if not self.predic_best:
                    self.ensemble_model.add_model(
                        self.model_selector.best_model)

                if self.update_hist:
                    self.model_performance_dict.update(self.model_selector.get_models_and_performance())

                self.select_and_update_pipelines()

                self.reset_exploration()

            self.EW_COUNTER += 1

        self._exploitation(x, y)
        self._exploration(x, y)


        self.COUNTER += 1

    def reset(self):
        self.__init__()


## Auto Stream Regression Code ##

class AutoStreamRegressor(base.Estimator):
    def __init__(
        self,
        metric=metrics.MAE(),
        prediction_mode="ensemble",
        ensemble_size=3,
        exploration_window=1000,
        budget=10,
        update_hist=False,
        return_metric=True,
        verbose=False,
        mode="greedy",
        no_fs_models=[],
        seed=42):
        """
        AutoStreamML class for Searching best online/streaming machine learning pipeline Combination.

        Parameters:
        - metric (object, optional): Evaluation metric for model performance (default: MAE).
        - prediction_mode (str, optional): 'ensemble' or 'best' (default: 'ensemble').
        - ensemble_size (int, optional): Maximum size of the dynamic ensemble (default: 3).
        - exploration_window (int, optional): Size of the exploration window (default: 1000).
        - budget (int, optional): Total budget for pipeline operations (default: 10).
        - update_hist (bool, optional): Whether to update the performance history (default: False).
        - return_metric (bool, optional): weather to return the performence metrics or not (default: True).
        - verbose (bool, optional): Whether to print verbose output (default: False).
        - mode (str, optional): 'greedy' or 'fast' (default: greedy)
        - no_fs_models (list,optional): Skip feature selection for listed models due to error or Not Needed (default: []).
        - seed (int, optional): Random seed for reproducibility (default: 42).
        """
        # Validate metric
        if not isinstance(metric, metrics.base.Metric):
            raise ValueError(f"The 'metric' parameter must be a valid metric object from sklearn.metrics. Received: {metric}, {type(metric)}")

        # Validate prediction_mode
        if prediction_mode not in ["ensemble", "best"]:
            raise ValueError(f"The 'prediction_mode' parameter must be 'ensemble' or 'best'. Received: {prediction_mode}, {type(prediction_mode)}")

        # Validate ensemble_size
        if not isinstance(ensemble_size, int) or ensemble_size < 1:
            raise ValueError(f"The 'ensemble_size' parameter must be a positive integer. Received: {ensemble_size}, {type(ensemble_size)}")

        # Validate exploration_window
        if not isinstance(exploration_window, int) or exploration_window < 1:
            raise ValueError(f"The 'exploration_window' parameter must be a positive integer. Received: {exploration_window}, {type(exploration_window)}")

        # Validate budget
        if not isinstance(budget, int) or budget < 1:
            raise ValueError(f"The 'budget' parameter must be a positive integer. Received: {budget}, {type(budget)}")

        # Validate update_hist
        if not isinstance(update_hist, bool):
            raise ValueError(f"The 'update_hist' parameter must be a boolean (True or False). Received: {update_hist}, {type(update_hist)}")

        # Validate return_metric
        if not isinstance(return_metric, bool):
            raise ValueError(f"The 'return_metric' parameter must be a boolean (True or False). Received: {return_metric}, {type(return_metric)}")

        # Validate verbose
        if not isinstance(verbose, bool):
            raise ValueError(f"The 'verbose' parameter must be a boolean (True or False). Received: {verbose}, {type(verbose)}")

        # Validate mode
        if mode not in ["greedy", "fast"]:
            raise ValueError(f"The 'mode' parameter must be 'greedy' or 'fast'. Received: {mode}, {type(mode)}")

        # Validate no_fs_models
        if not isinstance(no_fs_models, list):
            raise ValueError(f"The 'no_fs_models' parameter must be list of models names: {no_fs_models}, {type(no_fs_models)}")

        # Validate seed
        if not isinstance(seed, int):
            raise ValueError(f"The 'seed' parameter must be an integer. Received: {seed}, {type(seed)}")

        #Logger Init
        log_filename = "AutoStreamML.log"
        logging.basicConfig(filename=log_filename,level=logging.ERROR,format="[%(levelname)s] - %(message)s",force=True,filemode='w')

        self.metric = metric
        self.exploration_window = exploration_window
        self.verbose = verbose
        self.seed = seed
        self.COUNTER = 0
        self.EW_COUNTER = 0
        self.pipeline_list = []
        self.model = None
        self.ensemble_size = ensemble_size
        self.return_metric = return_metric
        self.mode = mode

        self.no_fs_models = no_fs_models

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.prediction_mode = prediction_mode

        self.update_hist = update_hist

        self.model_performance_dict = {}

        self.budget = budget

        # Allocate budget for the best pipeline (fixed value)
        self.best_pipeline_budegt = 1

        # Calculate the remaining budget after allocating to the best pipeline
        remaining_budget = self.budget - self.best_pipeline_budegt

        self.new_pipeline_budget = round(0.4 * remaining_budget)
        self.hyper_mutation_budegt = round(0.6 * remaining_budget)

        self.algorithms = []
        self.preprocessing_steps = []
        self.feature_selection_methods = []
        self.hyperparameters = hyperparameteers_options_reg

        self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in model_options_reg]
        self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in preprocessor_options_reg]
        self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in feature_selection_options_reg]

        self.ensemble_model = DynamicEnsembleRegressor(models=[],
                                                        max_size=self.ensemble_size,
                                                        mode=self.mode,
                                                        metric=self.metric.clone())

        self._create_pipelines()

        self.model_selector = ProgressiveModelSelector(models=self.pipeline_list,
                                                       verbose=False,
                                                       mode=self.mode,
                                                       metric=self.metric.clone())

        self.ensemble_model.add_model(self.model_selector.best_model)

        self.predic_best = False

        if self.prediction_mode != "ensemble":
            self.predic_best = True
            self.best = self.model_selector.best_model

    def _get_hyperparameters(self, model, hyper_init="random"):
        """
        Get hyperparameters for a algorithm.

        Parameters:
        - algorithm: The selected algorithm.

        Returns:
        - hyperparameters_with_algo: Hyperparameters for the algorithm.
        """
        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]

        hyperparams = {}

        if hyper_init == "random":
            for key, values in model_hyper.items():
                hyperparams[key] = random.choice(values)
        else:
            default_hyper = model._get_params()
            for key, values in model_hyper.items():
                if default_hyper[key] in values:
                    hyperparams[key] = default_hyper[key]
                else:
                    hyperparams[key] = values[0]

        return model._set_params(hyperparams)

    def _create_pipelines(self):
        """
        Create a list of machine learning pipelines combining preprocessing, algorithms, and feature selection.
        """
        for preprocessing_step in self.preprocessing_steps:
            for model_instance in self.algorithms:
                # Create a pipeline without feature selection
                pipeline = preprocessing_step | model_instance
                self.pipeline_list.append(pipeline.clone())

                if type(model_instance).__name__ not in self.no_fs_models:
                    # Create a pipeline with feature selection
                    for feature_selection in self.feature_selection_methods:
                        full_pipeline = preprocessing_step | feature_selection | model_instance
                        self.pipeline_list.append(full_pipeline.clone())

    def _initialize_random_pipeline(self):
        """
        Initialize a random machine learning pipeline.

        Returns:
        - random_pipeline: A randomly initialized pipeline.
        """

        # Ensure that each component type is selected at least once
        random_algorithm = random.choice(self.algorithms)
        random_preprocessing = random.choice(self.preprocessing_steps)
        random_feature_selection = random.choice(self.feature_selection_methods)

        new_choice = random.choice(["WITH_FEATURE", "WITHOUT_FEATURE"])

        if type(random_algorithm).__name__ in self.no_fs_models:
            new_choice = "WITHOUT_FEATURE"

        if new_choice == "WITH_FEATURE":
            random_feature_selection = random.choice(self.feature_selection_methods)
            return random_preprocessing | random_feature_selection | random_algorithm
        else:
            return random_preprocessing | random_algorithm

    def _get_current_params(self, model):
        """
        Get the current hyperparameters of a model.

        Parameters:
        - model: The machine learning model.

        Returns:
        - current_hyper: Current hyperparameters of the model.
        """

        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]
        current_hyper = {}
        for k, _ in model_hyper.items():
            current_hyper[k] = model._get_params()[k]
        return current_hyper

    def _ardns(self, current_value, values_list):
        """
        Adaptive Random Directed Nearby Search (ARDNS) function.

        Parameters
        ----------
        current_value : any
            The current value that you want to find a nearby option for.

        values_list : list
            A list of possible values to choose from.

        Returns
        -------
        any
            The nearby value selected based on the random directed nearby search strategy.
        """

        nearby_option = np.random.choice(["same", "upper", "lower", "random"])

        if nearby_option == "same":
            return current_value
        elif nearby_option == "upper":
            return values_list[min(values_list.index(current_value) + 1, len(values_list) - 1)]
        elif nearby_option == "lower":
            return values_list[max(values_list.index(current_value) - 1, 0)]
        else:  # "random"
            return np.random.choice(values_list)

    def _suggest_nearby_hyperparameters(self, model):
        """
        Suggest nearby hyperparameter values by perturbing the current values cyclically within a defined range.

        Parameters:
        - model: The machine learning model instance.

        Returns:
        - model: A copy of the model with suggested hyperparameter values.
        """

        # Get the current hyperparameter values
        current_hyperparameters = self._get_current_params(model)

        # Get the user define hyperparameter values
        user_defined_search_space = self.hyperparameters[type(model).__name__]
        # print(user_defined_search_space)

        suggested_hyperparameters = {}

        for param_name, param_value in current_hyperparameters.items():
            if isinstance(param_value, int):
                # Integer hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = int(new_value)

            elif isinstance(param_value, float):
                # Float hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = float(new_value)

            elif isinstance(param_value, (str, bool)):
                # Categorical hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = new_value

            else:
                # Unsupported type (e.g., NoneType)
                # random value selected from search space
                suggested_hyperparameters[param_name] = random.choice(user_defined_search_space[param_name])

        return model._set_params(suggested_hyperparameters)
        # return model.clone(new_params=suggested_hyperparameters)

    def next_nerby(self, pipeline):
        """
        Mutate a pipeline by perturbing hyperparameters.

        Parameters:
        - pipeline: The original machine learning pipeline.

        Returns:
        - next_nerby_pipeline: The mutated machine learning pipeline.
        """

        next_nerby_pipeline = list(pipeline.steps.values())

        new_algorithm = self._suggest_nearby_hyperparameters(next_nerby_pipeline[-1])
        new_preprocessing = self._suggest_nearby_hyperparameters(next_nerby_pipeline[0])

        if len(next_nerby_pipeline) > 2:
            new_feature_selection = self._suggest_nearby_hyperparameters(next_nerby_pipeline[1])
            return new_preprocessing | new_feature_selection | new_algorithm
        else:
            return new_preprocessing | new_algorithm

    def select_and_update_pipelines(self):
        """
        Select and update pipelines based on performance and budgets.
        """

        best_pipeline = self.best_model.clone()

        next_nerby_pipelines = []

        for _ in range(self.hyper_mutation_budegt):
            next_nerby_pipeline = self.next_nerby(best_pipeline)
            next_nerby_pipelines.append(next_nerby_pipeline)
            best_pipeline = next_nerby_pipeline.clone()

        if self.update_hist:
            new_pipelines = [self.select_models_probabilistically().clone() for i in range(self.new_pipeline_budget)]
        else:
            new_pipelines = [self._initialize_random_pipeline().clone() for i in range(self.new_pipeline_budget)]

        self.pipeline_list = [self.best_model.clone()] + next_nerby_pipelines + new_pipelines

        # maintaining uniquness
        self.pipeline_list = list(set(self.pipeline_list))

    def select_models_probabilistically(self):
        """
        Randomly select models with probabilities based on their historical performance scores.

        Returns:
        - selected_model: The selected model.
        """

        # Extract models and their performance scores from the dictionary
        models = list(self.model_performance_dict.keys())
        performance_scores = list(self.model_performance_dict.values())

        # Calculate the total performance score
        total_performance_score = sum(performance_scores)

        # Calculate the selection probabilities based on performance scores
        selection_probabilities = [score / total_performance_score for score in performance_scores]

        # Randomly select a model based on the calculated probabilities
        selected_model_idx = random.choices(range(len(models)), weights=selection_probabilities)[0]
        selected_model = models[selected_model_idx]

        return selected_model

    def reset_exploration(self):
        """
        Reset the exploration mechanism.
        """
        self.model_selector.update_models(self.pipeline_list)

    def print_batch_info(self):
        """
        Print information about the current batch and model performance.
        """
        print(
            f"Data Point: {self.COUNTER} | Exploration Window: {self.EW_COUNTER}")
        try:
            print(f"Best Pipeline: {self.best_model}")
            print(f"Best Preprocessor Hyper: {self._get_current_params(list(self.best_model.steps.values())[0])}")
            if len(list(self.best_model.steps.values())) == 3:
                print(f"Best Feature Hyper: {self._get_current_params(list(self.best_model.steps.values())[1])}")
            print(f"Best Model Hyper: {self._get_current_params(list(self.best_model.steps.values())[-1])}")
        except Exception as e:
            logging.error(f"[{self.COUNTER}] - [Print Section] - {e}")
        if self.return_metric:
            print(f"Best Pipeline Score: {self.metric.get():0.2f}%")
        print("----------------------------------------------------------------------")

    @property
    def best_model(self):
        """
        Get the best-performing model.

        Returns:
        - best_model: The best-performing machine learning model.
        """
        return self.model_selector.best_model

    def predict_one(self, x):
        """
        Make a prediction for a single data point.

        Parameters:
        - x: Input data.

        Returns:
        - y_pred: Predicted output.
        """
        try:
            if self.predic_best:
                try:
                    return self.best.predict_one(x)
                except Exception as e:
                    logging.error(f"[{self.COUNTER}] - [Predic Best Section] - {e}")
                    return None
            return self.ensemble_model.predict_one(x)
        except:
            logging.error(f"[{self.COUNTER}] - [Predic Ensemble Section] - {e}")
            return None

    def _exploration(self, x, y):
        """
        Perform exploration on incoming data.

        Parameters:
        - x: Input data.
        - y: Target labels.
        """
        # Exploration Time
        self.model_selector.learn_one(x, y)

    def _exploitation(self, x, y):
        """
        Perform exploitation on incoming data.

        Parameters:
        - x: Input data.S
        - y: Target labels.
        """
        # Exploitation Time
        if self.predic_best:
            if self.return_metric:
                try:
                    y_pred = self.best.predict_one(x)
                    self.metric.update(y, y_pred)
                    self.best.learn_one(x, y)
                except Exception as e:
                    logging.error(f"[{self.COUNTER}] - [Explotiation Section] - {e}")
        else:
            if self.return_metric:
                y_pred = self.ensemble_model.predict_one(x)
                self.metric.update(y, y_pred)
            self.ensemble_model.learn_one(x, y)

    def learn_one(self, x, y):
        """
        Learn from a single data point.

        Parameters:
        - x: Input data.
        - y: Target labels.
        """

        if self.COUNTER % self.exploration_window == 0:
            if self.predic_best:
                self.best = self.best_model

            if self.verbose:
                self.print_batch_info()

            if self.COUNTER != 0:
                if not self.predic_best:
                    self.ensemble_model.add_model(
                        self.model_selector.best_model)

                if self.update_hist:
                    self.model_performance_dict.update(self.model_selector.get_models_and_performance())

                self.select_and_update_pipelines()

                self.reset_exploration()

            self.EW_COUNTER += 1

        # try:
        self._exploitation(x, y)
        self._exploration(x, y)

        # except Exception as e:

        #     logging.error(f"[{self.COUNTER}] - [Learning Section] - {e}")

        self.COUNTER += 1

    def reset(self):
        self.__init__()