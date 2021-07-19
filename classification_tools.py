__author__ = "Lucas Ortega Venzel"
__license__ = "The Unlicense"
__version__ = "0.1"
__maintainer__ = "Lucas Ortega Venzel"
__email__ = "venzellucas@gmail.com"
__status__ = "Testing"


from itertools import product
from tqdm import tqdm
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score


class GridSearchTemporalValidation():
    
    def __init__(self, classificator, params, cv, scoring):
        """
            classificator : your initialized classificator.
            params : possible params combination in a dict.
            scoring : metric to control overfitting.
            needs_proba : boolean, False for predicted labels or True for predicted probabilities.
        """
        self.classificator = classificator
        self.params = [dict(zip(params.keys(), p)) for p in product(*params.values())]
        self.folds = cv
        self.scorer = scoring
        self.results_ = []
    
    def fit(self, X, y, X_valid, y_valid):
        """
            X : traning samples.
            y : traning labels.
            X_valid : validation samples.
            y_valid : validation labels.
        """
        param_combinations = self.params
        result = []
        for param in tqdm(param_combinations):
            self.classificator.set_params(**param, n_jobs=-1)
            
            score_train = cross_val_score(estimator=self.classificator, y=y, X=X, scoring=self.scorer, cv=5, n_jobs=-1)
            score_train = sum(score_train) / len(score_train)
            self.classificator.fit(X,y)
            score_valid = self.scorer(estimator=self.classificator, y_true=y_valid, X=X_valid)
            
            delta = abs(score_train - score_valid)
            result.append({**param,**{'Train':score_train, 'Test':score_valid, 'Delta':delta}})
            
        self.results_ = result