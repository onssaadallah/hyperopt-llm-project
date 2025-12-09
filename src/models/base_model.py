
class BaseModel:
    """Base model with fit, predict, and save methods."""
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
