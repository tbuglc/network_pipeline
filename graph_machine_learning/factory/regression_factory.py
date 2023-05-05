from constants.regression_models import RegressionModels
from models.regression.decision_tree_regression import DecisionTreeRegression
from models.regression.elastic_net_regression import ElasticNetRegression
from models.regression.laso_regression import LasoRegression
from models.regression.random_forest_regression import RandomForestRegression
from models.regression.ridge_regression import RidgeRegression
from models.regression.support_vector_regression import SupportVectorRegression
# from models.base.base_factory import BaseFactory


class RegressionFactory():
  # A static method that takes an integer `model` as input and returns a machine learning model based on the input value.
  @staticmethod
  def get_model(model=int):

    # Check the value of the input `model` and return the corresponding machine learning model.
    if model == RegressionModels.DECISION_TREE:
           print('DecisionTreeRegression')
           return DecisionTreeRegression()
    if model == RegressionModels.ELASTIC_NET:
           print('ElasticNetRegression')
           return ElasticNetRegression()
    if model == RegressionModels.LASO:
           print('LasoRegressione Model')
           return LasoRegression()
    if model == RegressionModels.RANDOM_FOREST:
           print('RandomForestRegression')
           return RandomForestRegression()
    if model == RegressionModels.RIDGE:
           print('RidgeRegressionl')
           return RidgeRegression()
    if model == RegressionModels.SVR:
           print('SupportVectorRegression')
           return SupportVectorRegression()
 
    # Return None if no matching model is found.
    print('== No model found ==')
    return None
