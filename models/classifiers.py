import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from .base import BaseModel
from .config import KNNConfig, SVMConfig, RFConfig, ETConfig, LRConfig

class KNNClassifier(BaseModel):
    def __init__(self, config: KNNConfig):
        super().__init__(config)
        self.model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            algorithm=config.algorithm,
            leaf_size=config.leaf_size,
            p=config.p,
            metric=config.metric,
            n_jobs=config.n_jobs
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class SVMClassifier(BaseModel):
    def __init__(self, config: SVMConfig):
        super().__init__(config)
        self.model = SVC(
            C=config.C,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            shrinking=config.shrinking,
            probability=config.probability, # Must be True for predict_proba
            tol=config.tol,
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            random_state=config.random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class RFClassifier(BaseModel):
    def __init__(self, config: RFConfig):
        super().__init__(config)
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            criterion=config.criterion,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            max_features=config.max_features,
            max_leaf_nodes=config.max_leaf_nodes,
            min_impurity_decrease=config.min_impurity_decrease,
            bootstrap=config.bootstrap,
            oob_score=config.oob_score,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            class_weight=config.class_weight,
            ccp_alpha=config.ccp_alpha
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class ETClassifier(BaseModel):
    def __init__(self, config: ETConfig):
        super().__init__(config)
        self.model = ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            criterion=config.criterion,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            max_features=config.max_features,
            max_leaf_nodes=config.max_leaf_nodes,
            min_impurity_decrease=config.min_impurity_decrease,
            bootstrap=config.bootstrap,
            oob_score=config.oob_score,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            class_weight=config.class_weight,
            ccp_alpha=config.ccp_alpha
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LRClassifier(BaseModel):
    def __init__(self, config: LRConfig):
        super().__init__(config)
        self.model = LogisticRegression(
            penalty=config.penalty,
            dual=config.dual,
            tol=config.tol,
            C=config.C,
            fit_intercept=config.fit_intercept,
            intercept_scaling=config.intercept_scaling,
            class_weight=config.class_weight,
            random_state=config.random_state,
            solver=config.solver,
            max_iter=config.max_iter,
            # multi_class is removed for better compatibility
            n_jobs=config.n_jobs,
            l1_ratio=config.l1_ratio
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
