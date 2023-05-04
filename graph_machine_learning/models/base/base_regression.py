from models.base.base_model import BaseModel
from sklearn.metrics import mean_squared_error

class BaseRegression(BaseModel):
    def error(self, t_val, t_pred):
       return mean_squared_error(t_val, t_pred) 