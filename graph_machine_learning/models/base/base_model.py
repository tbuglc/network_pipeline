from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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
        tuple: A tuple containing the predicted labels for the validation data and the er score of the model.
        """
        # Train the model on the training data
        self.fit(x_train=x_train, t_train=t_train)
        # Use the trained model to predict the labels of the validation data
        t_pred = self.predict(x_val=x_val)
        # Calculate the er score of the model on the validation data
        er = self.error(t_val=t_val, t_pred=t_pred)

        
        # Return the predicted labels and er score
        print('Error: ', er)
        
        return  self.model