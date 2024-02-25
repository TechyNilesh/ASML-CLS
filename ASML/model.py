from river import base,metrics
from .search import PipelineSearch
import random
import numpy as np
from collections import Counter



class AutoStreamClassifier(base.Classifier):
    def __init__(
        self,
        config_dict=None,
        metric=metrics.Accuracy(),
        exploration_window=1000,
        budget=10,
        ensemble_size=3,
        prediction_mode='ensemble', #best
        verbose=False,
        seed=42):
    
        self.metric = metric
        self.exploration_window = exploration_window
        self.budget = budget
        
        self.config_dict = config_dict
        
        self.COUNTER = 0
        
        self.verbose = verbose
        self.seed = seed
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        # Create an instance of PipelineSearch
        self.pipe_search = PipelineSearch(config_dict=self.config_dict,budget=self.budget-1)
        
        self.pipeline_list = self.pipe_search._create_pipelines()
        
        self._metrics = [type(self.metric)() for _ in range(len(self.pipeline_list))]
        
        self._best_model_idx = np.random.randint(len(self.pipeline_list))
        
        self.best_model = self.pipeline_list[self._best_model_idx]
        
        self.prediction_mode = prediction_mode
        
        if self.prediction_mode=='ensemble':
        
            self.ensemble_size = ensemble_size

            self.model_snapshots = [self.pipeline_list[np.random.randint(len(self.pipeline_list))] for _ in range(self.ensemble_size)]#[self.pipeline_list[self._best_model_idx]]

            self.model_snapshots_metrics = [type(self.metric)() for _ in range(self.ensemble_size)]#[type(self.metric)()]
        

    def reset_exploration(self):
        
        self._metrics = [type(self.metric)() for _ in range(len(self.pipeline_list))]
        self._best_model_idx = np.random.randint(len(self.pipeline_list))
        if self.prediction_mode=='ensemble':
            self.model_snapshots_metrics = [type(self.metric)() for _ in range(len(self.model_snapshots))]
        
    def print_batch_info(self):
        print(
            f"Data Point: {self.COUNTER}")
        try:
            print(f"Best Pipeline: {self.best_model}")
            print(f"Best Preprocessor Hyper: {self.pipe_search._get_current_params(list(self.best_model.steps.values())[0])}")
            if len(list(self.best_model.steps.values())) == 3:
                print(f"Best Feature Hyper: {self.pipe_search._get_current_params(list(self.best_model.steps.values())[1])}")
            print(f"Best Model Hyper: {self.pipe_search._get_current_params(list(self.best_model.steps.values())[-1])}")
        except Exception as e:
            pass
        print("----------------------------------------------------------------------")
    
    def predict_one(self, x):
        
        if self.prediction_mode=='ensemble':
            
            votes = []
        
            for clf in self.model_snapshots:
                try:
                    vote = clf.predict_proba_one(x)
                    votes.append(vote)
                except Exception as e:
                    pass

            agg = Counter()

            for vote in votes:
                agg.update(vote)

            # Get the class with the highest weighted probability
            return agg.most_common(1)[0][0] if agg else None
        else: 
            try:
                return self.best_model.predict_one(x)
            except:
                return None
        
    def are_models_equal(self, model1, model2):
        # Check if the models have the same class name
        if model1.__class__.__name__ != model2.__class__.__name__:
            return False

        # Check if the models have the same hyperparameters
        if model1._get_params() != model2._get_params():
            return False

        return True

    def learn_one(self, x, y):
        
        # Update and train the best model and pipeline list
        for idx, _ in enumerate(self.pipeline_list):
            
            try:
                y_pred = self.pipeline_list[idx].predict_one(x)
                self._metrics[idx].update(y, y_pred)
                self.pipeline_list[idx].learn_one(x, y)

                # Check for a new best model
                if self._metrics[idx].is_better_than(self._metrics[self._best_model_idx]):
                    self._best_model_idx = idx
            
            except Exception as e:
                pass
                # Optionally handle exceptions here
        
        if self.prediction_mode=='ensemble':
            for idx, _ in enumerate(self.model_snapshots):
                try:
                    y_pred = self.model_snapshots[idx].predict_one(x)
                    self.model_snapshots_metrics[idx].update(y, y_pred)
                    self.model_snapshots[idx].learn_one(x,y)
                except:
                    pass
        else:
            try:
                self.best_model.learn_one(x,y)
                #print("Best Model")
            except:
                pass

        self.COUNTER += 1
        self._check_exploration_phase()


    def _check_exploration_phase(self):
        
        if self.COUNTER % self.exploration_window == 0:
            
            #worst_idx = np.argmin([m.get() for m in self.model_snapshots_metrics])
            
            #model_changed = not self.are_models_equal(self.pipeline_list[self._best_model_idx], self.best_model)
            #model_changed = self._metrics[self._best_model_idx].is_better_than(self.model_snapshots_metrics[worst_idx]):
            
            #if model_changed:
                                        
            self.best_model = self.pipeline_list[self._best_model_idx]
            
            if self.prediction_mode=='ensemble':
                # Limit the number of snapshots to avoid memory issues
                if len(self.model_snapshots) >= self.ensemble_size:
                    worst_idx = np.argmin([m.get() for m in self.model_snapshots_metrics])
                    self.model_snapshots.pop(worst_idx)
                    self.model_snapshots_metrics.pop(worst_idx)

                self.model_snapshots.append(self.best_model)
                self.model_snapshots_metrics.append(type(self.metric)())

            #print("Model Changed!")
            
            if self.verbose:
                self.print_batch_info()
            
            # Pass the model_changed flag to the select_and_update_pipelines method
            self.pipeline_list = self.pipe_search.select_and_update_pipelines(self.best_model)
            self.reset_exploration()     

    def reset(self):
        self.__init__()