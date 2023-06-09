from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, cross_validate
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self) -> None:
        # Initialize the instance variables
        self.model = None
        self.scaler = StandardScaler()
        
    @abstractmethod
    def error(self, t_val, t_pred):
        pass
        

    def fit(self, x_train, t_train):
        """
        Train the machine learning model on the training data.

        Parameters:
        x_train (numpy.ndarray): The input features of the training data.
        t_train (numpy.ndarray): The corresponding labels of the training data.

        Returns:
        None
        """
        # Scale the input features
        x_train_std = self.scaler.fit_transform(x_train)
        # Fit the scaled data and labels to the model
        self.model.fit(x_train_std, t_train)

    def predict(self, x_val):
        """
        Use the machine learning model to predict labels for new data.

        Parameters:
        x_val (numpy.ndarray): The input features of the validation data.

        Returns:
        numpy.ndarray: The predicted labels for the validation data.
        """
        # Scale the input features
        x_val_std = self.scaler.transform(x_val)
        # Use the model to predict the labels
        return self.model.predict(x_val_std)

    
    def params_tuning_with_cross_validation(self, x_train, t_train, param_grid, cv):
        """
        Perform hyperparameter tuning using cross-validation.

        Parameters:
        x_train (numpy.ndarray): The input features of the training data.
        t_train (numpy.ndarray): The corresponding labels of the training data.
        param_grid (dict): The hyperparameter space to search over.
        cv (int): The number of cross-validation folds.

        Returns:
        float: The best score obtained during cross-validation.
        """
        # Use GridSearchCV to search over the hyperparameter space
        grid_search = GridSearchCV(self.model, param_grid=param_grid, verbose=1, n_jobs=-1, refit=True, return_train_score=True, cv=cv)
        grid_search.fit(x_train, t_train)
        # Set the model to the best estimator found
        self.model = grid_search.best_estimator_
        # Return the best score
        return grid_search.best_score_,

    def train(self, x_train, x_val, t_train, t_val):
        """
        Train the machine learning model on the training data and evaluate it on the validation data.

        Parameters:
        x_train (numpy.ndarray): The input features of the training data.
        x_val (numpy.ndarray): The input features of the validation data.
        t_train (numpy.ndarray): The corresponding labels of the training data.
        t_val (numpy.ndarray): The true labels of the validation data.

        Returns:
        tuple: A tuple containing the predicted labels for the validation data and the err score of the model.
        """
        # Train the model on the training data
        self.fit(x_train=x_train, t_train=t_train)
        # Use the trained model to predict the labels of the validation data
        t_pred = self.predict(x_val=x_val)
        # print(t_pred)
        # print(np.concatenate([t_pred, t_val], axis=1))
        d = pd.DataFrame(np.concatenate([t_val, np.ones(t_val.shape),t_pred], axis=1))

     
        d.to_csv('error_diff.csv')
        # Calculate the error score of the model on the validation data
        err = self.error(t_val=t_val, t_pred=t_pred)
        # Return the predicted labels and error score
        print('err: ', err)
        
        return  self.model
    
    def cross_validate_model_performance(self, x_train,  t_train, cv=10, scroring=None):
        scores = cross_val_score(self.model, x_train, t_train, scoring=scroring, cv=cv)

        print('Cross-validation scores: ', scores)

        print("Mean accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores)))

        return scores
    
    def cross_validate_prediction(self, x_val, t_val, cv=10):
        predictions = cross_val_predict(self.model, x_val, t_val, cv=cv)

        print('\n CROSS VALIDATION PREDICTIONS \n')
        print(predictions)

        return predictions
    
    def cross_validate_multi_scores(self, x_train, t_train, cv=10, scoring=None):
        results = cross_validate(self.model, x_train, t_train, scoring=scoring, cv=5)
        
        print("Cross-validation results:")
        for metric in results.keys():
            print(metric, ": ", results[metric])
        
        return results