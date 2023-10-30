from river import base,tree
from river import metrics
import numpy as np
from collections import Counter
#from collections import deque
import logging


class DynamicEnsembleClassifier(base.Classifier):

    def __init__(self, models=[], max_size=3,metric=metrics.Accuracy(),use_probabilities=True,mode='greedy'):
        """
        Initialize a dynamic ensemble classifier.

        Parameters:
        - models (list): List of models to start with (default is an empty list).
        - max_size (int): Maximum number of models in the ensemble (default is 3).
        """
        self.models = models
        self.max_size = max_size
        self.metric = metric
        self.metrics = []
        self.use_probabilities = use_probabilities
        self.mode=mode # Option:['greedy','fast']
        self.current_model_index = 0

        if len(self.models) != 0:
            self.metrics = [self.metric.clone() for _ in range(max_size)]  # Store metrics for each model
     

    def predict_one(self, x):
        """
        Predict the class label for a single input instance.

        Parameters:
        - x: Input instance.

        Returns:
        - prediction: Predicted class label.
        """
        if self.use_probabilities:
            votes = [clf.predict_proba_one(x) for clf in self.models]
        else:
            votes = [{clf.predict_one(x):1}for clf in self.models]

        agg = Counter()

        for vote in votes:
            agg.update(vote)

        # Get the class with the highest weighted probability
        return agg.most_common(1)[0][0] if agg else None

    def learn_one(self, x, y):
        """
        Update the ensemble with a single instance.

        Parameters:
        - x: Input instance.
        - y: True class label.
        """
        if self.mode=='greedy':
            for idx in range(len(self.models)):
                y_pred = self.models[idx].predict_one(x)
                self.metrics[idx].update(y,y_pred)
                self.models[idx].learn_one(x,y)
                
        else:
            y_pred = self.models[self.current_model_index].predict_one(x)
            self.metrics[self.current_model_index].update(y,y_pred)
            self.models[self.current_model_index].learn_one(x,y)

            # Update the current model index in a round-robin fashion
            self.current_model_index = (self.current_model_index + 1) % len(self.models)

    def add_model(self, new_model):
        """
        Add a new model to the ensemble.

        Parameters:
        - new_model: New model to be added.
        """
        # Check if the new model is not already in the list
        if new_model not in self.models:
            # Check if the number of models exceeds the limit
            if self.is_full():
                # Find the weakest performing model based on metrics
                weakest_index = np.argmin([i.get() for i in self.metrics])
                
                # Remove the weakest model and its associated metric
                self.models.pop(weakest_index)
                self.metrics.pop(weakest_index)

            # Add the new model and create a new metric for it
            self.models.append(new_model)
            self.metrics.append(self.metric.clone())  # Replace with the appropriate metric

    def is_full(self):
        """
        Check if the ensemble is full (i.e., contains the maximum number of models).

        Returns:
        - is_full: True if the ensemble is full, False otherwise.
        """
        return len(self.models) >= self.max_size
    
    def del_model(self,index):
        del self.models[index]
        del self.metrics[index]
    
    def reset(self):
        """
        Clear all models in the ensemble and reset the trainable flags.
        """
        if len(self.models)>=1:
            self.models = [clf.clone() for clf in self.models]
        if len(self.metrics)>=1:
            self.metrics = [metric.clone() for metric in self.metrics]
        print("Ensembel Model is Reseted!")

class DynamicEnsembleRegressor(base.Regressor):

    def __init__(self, models=[], max_size=3, metric=metrics.MAE(), mode='greedy'):
        """
        Initialize a dynamic ensemble regressor.

        Parameters:
        - models (list): List of models to start with (default is an empty list).
        - max_size (int): Maximum number of models in the ensemble (default is 3).
        """
        self.models = models
        self.max_size = max_size
        self.metric = metric
        self.metrics = []
        self.mode = mode  # Option:['greedy','fast']
        self.current_model_index = 0

        if len(self.models) != 0:
            self.metrics = [self.metric.clone() for _ in range(max_size)]  # Store metrics for each model

    def predict_one(self, x):
        """
        Predict the target value for a single input instance.

        Parameters:
        - x: Input instance.

        Returns:
        - prediction: Predicted target value.
        """
        predictions = []
        for reg in self.models:
            predictions.append(reg.predict_one(x))

        # Aggregate predictions
        prediction = np.mean(predictions)

        return prediction

    def learn_one(self, x, y):
        """
        Update the ensemble with a single instance.

        Parameters:
        - x: Input instance.
        - y: True target value.
        """
        if self.mode == 'greedy':
            for idx in range(len(self.models)):
                y_pred = self.models[idx].predict_one(x)
                self.metrics[idx].update(y, y_pred)
                self.models[idx].learn_one(x, y)

        else:
            y_pred = self.models[self.current_model_index].predict_one(x)
            self.metrics[self.current_model_index].update(y, y_pred)
            self.models[self.current_model_index].learn_one(x, y)

            # Update the current model index in a round-robin fashion
            self.current_model_index = (self.current_model_index + 1) % len(self.models)

    def add_model(self, new_model):
        """
        Add a new model to the ensemble.

        Parameters:
        - new_model: New model to be added.
        """
        # Check if the new model is not already in the list
        if new_model not in self.models:
            # Check if the number of models exceeds the limit
            if self.is_full():
                # Find the weakest performing model based on metrics
                weakest_index = np.argmax([i.get() for i in self.metrics])

                # Remove the weakest model and its associated metric
                self.models.pop(weakest_index)
                self.metrics.pop(weakest_index)

            # Add the new model and create a new metric for it
            self.models.append(new_model)
            self.metrics.append(self.metric.clone())  # Replace with the appropriate metric

    def is_full(self):
        """
        Check if the ensemble is full (i.e., contains the maximum number of models).

        Returns:
        - is_full: True if the ensemble is full, False otherwise.
        """
        return len(self.models) >= self.max_size

    def delete_model(self, index):
        """
        Delete a model from the ensemble.

        Parameters:
        - index: Index of the model to be deleted.
        """
        if 0 <= index < len(self.models):
            self.models.pop(index)
            self.metrics.pop(index)

    def reset(self):
        """
        Clear all models in the ensemble and reset the trainable flags.
        """
        if len(self.models) >= 1:
            self.models = [reg.clone() for reg in self.models]
        if len(self.metrics) >= 1:
            self.metrics = [metric.clone() for metric in self.metrics]

class ProgressiveModelSelector(base.Estimator):
    
    def __init__(self, models, verbose=False,metric=metrics.CrossEntropy(),mode='greedy'):
        """
        Initialize a progressive model selector.

        Parameters:
        - models (list): List of models to select from.
        - verbose (bool): Whether to print information during the selection process (default is False).
        """
        self.models = models
        self.metric = metric

        self._metrics = [self.metric.clone() for _ in range(len(self.models))]

        self.verbose = verbose

        self.current_model_index = 0  # Initialize the index of the current model

        self._best_model_idx = np.random.randint(len(self.models))#0

        self.mode=mode # Option:['greedy','fast']

        #Logger Init
        log_filename = "ProgressiveModelSelector.log"
        logging.basicConfig(filename=log_filename,level=logging.ERROR,format="[%(levelname)s] - %(message)s",force=True,filemode='w')

    def learn_one(self, x, y):
        """
        Update the model selector with a single data point.

        Parameters:
        - x: Input data point.
        - y: True target value.
        """
        if self.mode=='greedy':
            for idx in range(len(self.models)):
                try:
                    y_pred = self.models[idx].predict_one(x)
                    self._metrics[idx].update(y, y_pred)
                    self.models[idx].learn_one(x, y)
                except Exception as e:
                    logging.error(f"An error occurred during 'predict_one' or 'learn_one' in mode 'greedy': {idx}")
                # Check for a new best model
                if  self._metrics[idx].is_better_than(self._metrics[self._best_model_idx]):
                    self._best_model_idx = idx
        else:

            # Select the current model based on the current index
            current_model = self.models[self.current_model_index]

            current_metric = self._metrics[self.current_model_index]

            try:
                # Make predictions with the updated model
                y_pred = current_model.predict_one(x)
                # Update the score
                current_metric.update(y, y_pred)
                # Learn from the new data point
                current_model.learn_one(x, y)
            except:
                logging.error(f"An error occurred during 'predict_one' or 'learn_one' in mode 'fast': {idx}")
        

            # Check for a new best model
            if current_metric.is_better_than(self._metrics[self._best_model_idx]):
                self._best_model_idx = self.current_model_index

            # Update the current model index in a round-robin fashion
            self.current_model_index = (self.current_model_index + 1) % len(self.models)

        if self.verbose:
            self._print()

    def predict_one(self, x):
        """
        Predict using the best model selected so far.

        Parameters:
        - x: Input data point.

        Returns:
        - prediction: Predicted value.
        """
        # Predict using the best model (the one with the highest cumulative score)
        return self.best_model.predict_one(x)

    def _print(self):
        """
        Print information about the best model and its score.
        """
        best_model = self.best_model
        best_score = self._metrics[self._best_model_idx]
        print(f"Best Model: {best_model}, Score: {best_score}")

    @property
    def best_model(self):
        """
        Get the best model selected so far.

        Returns:
        - best_model: Best model.
        """
        return self.models[self._best_model_idx]

    def get_models_and_performance(self):
        """
        Returns a dictionary of models and their performance scores.
        """
        model_performance_dict = {}
        for idx, model in enumerate(self.models):
            model_performance = self._metrics[idx].get()
            model_performance_dict[model] = model_performance
        return model_performance_dict
    
    def get_top_n_models(self,n):
        m_l = [m.get() for m in self._metrics]
        return [self.models[i] for i in np.argsort(m_l)[::-1][:n]]
    
    def update_models(self,new_models):
        self.models = new_models
        self._metrics = [self.metric.clone() for _ in range(len(self.models))]
        self.current_model_index = 0
        self._best_model_idx = 0

    def reset(self):
        """
        Clear all models in the ensemble and reset the trainable flags.
        """
        if len(self.models) >= 1:
            self.models = [model.clone() for model in self.models]
        if len(self._metrics) >= 1:
            self._metrics = [metric.clone() for metric in self._metrics]