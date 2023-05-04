from sklearn.tree import DecisionTreeRegressor
import numpy as np

from models.base.base_regression import BaseRegression

class DecisionTreeRegression(BaseRegression):
    def __init__(self) -> None:
        super().__init__()
        self.model = DecisionTreeRegressor()