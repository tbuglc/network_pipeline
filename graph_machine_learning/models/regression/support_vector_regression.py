from sklearn.svm import SVR
import numpy as np

from models.base.base_regression import BaseRegression

class SupportVectorRegression(BaseRegression):
    def __init__(self) -> None:
        super().__init__()
        
        self.model = SVR()