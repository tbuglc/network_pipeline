from models.base.base_mode import BaseModel
from sklearn.metrics import accuracy_score

class BaseClassification(BaseModel):
    def error(self, t_val, t_pred):
       return accuracy_score(t_val, t_pred) 