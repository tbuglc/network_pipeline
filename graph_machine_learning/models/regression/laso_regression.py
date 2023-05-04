from sklearn.linear_model import Lasso
from models.base.base_regression import BaseRegression

class LasoRegression(BaseRegression):
    def __init__(self) -> None:
        super().__init__()
        self.model = Lasso(alpha=1.0)