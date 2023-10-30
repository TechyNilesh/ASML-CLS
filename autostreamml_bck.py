import random
import numpy as np
from river import metrics
from river import utils
from river import base
from .models import ProgressiveModelSelector,DynamicEnsembleClassifier,DynamicEnsembleRegressor
from .clf_config import model_options,preprocessor_options,feature_selection_options,hyperparameters_options
from .reg_config import model_options_reg,preprocessor_options_reg,feature_selection_options_reg,hyperparameteers_options_reg



    
class AutoStreamML(base.Estimator):
    
    def __init__(self, 
                 metric=metrics.Accuracy(),
                 task='cls',
                 prediction_mode = 'ensemble',
                 ensemble_size = 3,
                 exploration_window=1000,
                 budget=10,
                 update_hist=False,
                 return_metric = True,
                 verbose=False,
                 mode = 'greedy',
                 seed=42):
        """
        AutoStreamML class for Searching best online/streaming machine learning pipeline Combination.

        Parameters:
        - metric (object, optional): Evaluation metric for model performance (default: Accuracy).
        - task (str, optional): 'cls' for classification or 'reg' for regression (default: 'cls').
        - prediction_mode (str, optional): 'ensemble' or 'best' (default: 'ensemble').
        - ensemble_size (int, optional): Maximum size of the dynamic ensemble (default: 3).
        - exploration_window (int, optional): Size of the exploration window (default: 1000).
        - budget (int, optional): Total budget for pipeline operations (default: 10).
        - update_hist (bool, optional): Whether to update the performance history (default: False).
        - return_metric (bool, optional): weather to return the performence metrics or not (default: True).
        - verbose (bool, optional): Whether to print verbose output (default: False).
        - mode (str, optional): 'greedy' or 'fast' (default: greedy)
        - seed (int, optional): Random seed for reproducibility (default: 42).
        """
        # Validate metric
        if not isinstance(metric, metrics.base.Metric):
            raise ValueError(f"The 'metric' parameter must be a valid metric object from sklearn.metrics. Received: {metric}, {type(metric)}")
        
        # Validate task
        if task not in ['cls', 'reg']:
            raise ValueError(f"The 'task' parameter must be 'cls' for classification or 'reg' for regression. Received: {task}, {type(task)}")
        
        # Validate prediction_mode
        if prediction_mode not in ['ensemble', 'best']:
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
        if mode not in ['greedy', 'fast']:
            raise ValueError(f"The 'mode' parameter must be 'greedy' or 'fast'. Received: {mode}, {type(mode)}")
        
        # Validate seed
        if not isinstance(seed, int):
            raise ValueError(f"The 'seed' parameter must be an integer. Received: {seed}, {type(seed)}")

        
        self.metric = metric
        self.exploration_window = exploration_window
        self.verbose = verbose
        self.seed = seed
        self.COUNTER = 0
        self.EW_COUNTER = 0
        self.pipeline_list = []
        self.task=task
        self.model=None
        self.ensemble_size = ensemble_size
        self.return_metric = return_metric
        self.mode = mode

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.prediction_mode = prediction_mode
                
        self.update_hist = update_hist
        
        self.model_performance_dict = {}
        
        self.budget = budget
        
        self.new_pipeline_budget = int(self.budget*0.40)
        self.hyper_mutation_budegt = int(self.budget*0.50)
        self.best_pipeline_budegt = int(self.budget*0.10)

        if self.task=='cls':
            self.algorithms = model_options
            self.preprocessing_steps = preprocessor_options 
            self.feature_selection_methods = feature_selection_options
            self.hyperparameters = hyperparameters_options
            self.ensemble_model = DynamicEnsembleClassifier(max_size=self.ensemble_size,mode=self.mode,metric=self.metric.clone())
        else:
            self.algorithms = model_options_reg
            self.preprocessing_steps = preprocessor_options_reg
            self.feature_selection_methods = feature_selection_options_reg
            self.hyperparameters = hyperparameteers_options_reg
            self.ensemble_model = DynamicEnsembleRegressor(max_size=self.ensemble_size,mode=self.mode,metric=self.metric.clone())
        
        self._create_pipelines()

        self.pms = ProgressiveModelSelector(models=self.pipeline_list,verbose=False,mode=self.mode,metric=self.metric.clone())
        
        self.ensemble_model.add_model(self.pms.best_model)

        self.predic_best = False
        
        if self.prediction_mode!="ensemble":
            self.predic_best = True
            self.best = self.pms.best_model

    def _create_pipelines(self):
        """
        Create a list of machine learning pipelines combining preprocessing, algorithms, and feature selection.
        """
        for preprocessing_step in self.preprocessing_steps:
            for model_instance in self.algorithms:
                # Create a pipeline without feature selection
                pipeline = preprocessing_step | model_instance
                self.pipeline_list.append(pipeline)
                
                if self.task=='cls':
                    # Create a pipeline with feature selection
                    for feature_selection in self.feature_selection_methods:
                        full_pipeline = preprocessing_step | feature_selection | model_instance
                        self.pipeline_list.append(full_pipeline)

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
        
        if self.task=='cls':
            new_choice = random.choice(['WITH_FEATURE','WITHOUT_FEATURE'])
        else:
            new_choice = 'WITHOUT_FEATURE'

        if new_choice=='WITH_FEATURE':
            random_feature_selection = random.choice(self.feature_selection_methods)
            random_pipeline = random_preprocessing | random_feature_selection | random_algorithm
        else:
            random_pipeline = random_preprocessing | random_algorithm

        return random_pipeline
    
    def _get_hyperparameters(self,random_algorithm):
        """
        Get hyperparameters for a random algorithm.

        Parameters:
        - random_algorithm: The randomly selected algorithm.

        Returns:
        - random_hyperparameters_with_algo: Hyperparameters for the algorithm.
        """
        
        model_name = type(random_algorithm).__name__
        model_hyper = self.hyperparameters[model_name]
        try:
            hyperparameters = utils.expand_param_grid(random_algorithm, model_hyper)
            random_hyperparameters_with_algo = random.choice(hyperparameters)
        except:
            random_hyperparameters_with_algo = random_algorithm
        
        return random_hyperparameters_with_algo
    
    def _get_current_params(self,model):
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
        for k,_ in model_hyper.items():
            current_hyper[k] = model._get_params()[k]
        return current_hyper
    
    def _circular_traverse(self,current_value, my_list):
        """
        Traverse a list in a circular manner.

        Parameters:
        - current_value: Current value in the list.
        - my_list: The list to traverse.

        Returns:
        - next_value: The next value to visit in the list.
        """
                
        # Find the index of the current value in the list
        try:
            current_index = my_list.index(current_value)
        except ValueError:
            # Handle the case where the current value is not in the list
            current_index = 0

        # Calculate the index of the next value in a circular manner
        next_index = (current_index + 1) % len(my_list)

        # Return the next value to visit
        return my_list[next_index]

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
        
        #print("Current HyperParameter",current_hyperparameters)

        #Get the user define hyperparameter values
        user_defined_search_space = self.hyperparameters[type(model).__name__]

        suggested_hyperparameters = {}

        for param_name, param_value in current_hyperparameters.items():

            if isinstance(param_value, int):
                # Integer hyperparameter
                perturbed_value = self._circular_traverse(param_value,user_defined_search_space[param_name])

                suggested_hyperparameters[param_name]=perturbed_value

            elif isinstance(param_value, float):
                # Float hyperparameter
                perturbed_value = self._circular_traverse(param_value,user_defined_search_space[param_name])

                suggested_hyperparameters[param_name]=perturbed_value

            elif isinstance(param_value, (str, bool)):
                # Categorical hyperparameter
                perturbed_value = self._circular_traverse(param_value,user_defined_search_space[param_name])

                suggested_hyperparameters[param_name]=perturbed_value

            else:
                # Unsupported type (e.g., NoneType)
                # Keep the current value unchanged
                suggested_hyperparameters[param_name]= random.choice(user_defined_search_space[param_name])
        
        #print("Suggested HyperParameter",suggested_hyperparameters)
        try:
            return model._set_params(suggested_hyperparameters)
        except:
            return model.clone(new_params=suggested_hyperparameters)
    
    def mutate_pipeline(self, pipeline):
        """
        Mutate a pipeline by perturbing hyperparameters.

        Parameters:
        - pipeline: The original machine learning pipeline.

        Returns:
        - mutated_pipeline: The mutated machine learning pipeline.
        """
        
        mutated_pipeline = list(pipeline.steps.values())
        
        new_algorithm = self._suggest_nearby_hyperparameters(mutated_pipeline[-1])
        new_preprocessing = self._suggest_nearby_hyperparameters(mutated_pipeline[0])

        if len(mutated_pipeline)>2:
            new_feature_selection = self._suggest_nearby_hyperparameters(mutated_pipeline[1])
            return new_preprocessing | new_feature_selection | new_algorithm
        else:
            return new_preprocessing | new_algorithm

    def select_and_update_pipelines(self):
        """
        Select and update pipelines based on performance and budgets.
        """
        
        best_pipeline = self.best_model.clone()

        mutated_pipelines = []

        for _ in range(self.hyper_mutation_budegt):
            mutated_pipeline = self.mutate_pipeline(best_pipeline)
            mutated_pipelines.append(mutated_pipeline)
            best_pipeline = mutated_pipeline.clone()
        
        if self.update_hist:
            new_pipelines = [self.select_models_probabilistically().clone() for i in range(self.new_pipeline_budget)]
        else:
            new_pipelines = [self._initialize_random_pipeline().clone() for i in range(self.new_pipeline_budget)]

        self.pipeline_list = [self.best_model.clone()]*self.best_pipeline_budegt + mutated_pipelines + new_pipelines 
        #self.pipeline_list = mutated_pipelines + new_pipelines 

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
        selection_probabilities = [
            score / total_performance_score
            for score in performance_scores
        ]

        # Randomly select a model based on the calculated probabilities
        selected_model_idx = random.choices(range(len(models)), weights=selection_probabilities)[0]
        selected_model = models[selected_model_idx]

        return selected_model

    def reset_exploration(self):
        """
        Reset the exploration mechanism.
        """
        self.pms.update_models(self.pipeline_list)

        
    def print_batch_info(self):
        """
        Print information about the current batch and model performance.
        """
        print(f"Data Point: {self.COUNTER} | Exploration Window: {self.EW_COUNTER}")
        try:
            print(f"Best Pipeline: {self.best_model}")
            #print(f"Best Model Hyper: {self._get_current_params(list(self.best_model.steps.values())[-1])}")
        except Exception as e:
            print(e)
        if self.return_metric:
            if self.task=='cls':
                print(f"Best Pipeline Score: {self.metric.get() * 100:0.2f}%")
            else:
                print(f"Best Pipeline Score: {round(self.metric.get(),2)}")
        print("----------------------------------------------------------------------")

    @property
    def best_model(self):
        """
        Get the best-performing model.

        Returns:
        - best_model: The best-performing machine learning model.
        """
        return self.pms.best_model
    
    def predict_one(self,x):
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
                except:
                    return None
            return self.ensemble_model.predict_one(x)
        except:
            return None

    def _exploration(self,x,y):

        """
        Perform exploration on incoming data.

        Parameters:
        - x: Input data.
        - y: Target labels.
        """
        #Exploration Time
        self.pms.learn_one(x, y)
    
    def _exploitation(self,x,y):
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
                    self.metric.update(y,y_pred)
                    self.best.learn_one(x,y)
                except:
                    print("Error in Explotiation")
            
        else:
            if self.return_metric:
                y_pred = self.ensemble_model.predict_one(x)
                self.metric.update(y,y_pred)
            self.ensemble_model.learn_one(x,y)

    def learn_one(self, x, y):

        """
        Learn from a single data point.

        Parameters:
        - x: Input data.
        - y: Target labels.
        """

        if self.COUNTER<self.exploration_window:
                if self.COUNTER % (self.exploration_window/self.ensemble_size)==0:
                    self.ensemble_model.add_model(self.pms.best_model)
                    self.best=self.best_model
        
        if self.COUNTER % self.exploration_window == 0:
            
            if self.predic_best:
                self.best = self.best_model
                #print("prediction_model:",self.best)
            
            if self.verbose:
                self.print_batch_info()
                #print(self.pms._metrics[self.pms._best_model_idx].get())
            
            if self.COUNTER!=0:
                
                self.ensemble_model.add_model(self.pms.best_model)
                
                if self.update_hist:
                    self.model_performance_dict.update(self.pms.get_models_and_performance())
                
                self.select_and_update_pipelines()
                
                self.reset_exploration()
            
            self.EW_COUNTER += 1
        
        self._exploitation(x,y)
        self._exploration(x,y)
        
        self.COUNTER += 1

    def reset(self):
        self.__init__()
