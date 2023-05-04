import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from models.base.base_model import BaseModel


class SVM(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC()
    # override 
    def classify(self, x_train, x_val, t_train, t_val):
        # Train the classifier on the training set
        t_train = np.argmax(t_train, axis=1)
        t_val = np.argmax(t_val, axis=1)
        
        return super().classify(x_train=x_train, x_val=x_val, t_train=t_train,t_val=t_val)
        
    