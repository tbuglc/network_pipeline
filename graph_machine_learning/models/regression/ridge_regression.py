from sklearn.linear_model import Ridge
import numpy as np

from models.base.base_regression import BaseRegression

class RidgeRegression(BaseRegression):
    def __init__(self) -> None:
        super().__init__()
        self.model = Ridge(alpha=1.0)