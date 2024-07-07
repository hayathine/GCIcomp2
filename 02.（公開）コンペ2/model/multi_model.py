from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.early_stop import no_progress_loss
from .base import Base_model


class Lgbm_hyperopt(Base_model):
    def __init__(self, X_train, y_train, X_valid, y_valid, params, n_iter, random_state = 0, max_depth = 3):
        super().__init__(X_train, y_train, X_valid, y_valid, params, n_iter, random_state, )
        self.max_depth = max_depth

    def lgb_objective(self, args, ):
        lgb = LGBMClassifier(
            num_leaves = args['num_leaves'],
            max_depth = self.max_depth,
            n_estimators = args['n_estimators'],
            learning_rate = args['learning_rate'],
            # min_child_samples = args['min_child_samples'],
            reg_lambda = args['reg_lamb'],
            bagging_freq = args['bagging_freq'],
            bagging_fraction = args['bagging_fraction'],
            feature_fraction = args['feature_fraction'],
            min_data_in_leaf = args['min_data_in_leaf'],
            # subsample_freq = args['sub_sample_freq'],
            # subsample = args['sub_sample'],
            random_state = self.random_state,
            objective = self.space['objective'],
            metric = self.space['metric'],
            verbosity = self.space['verbosity'],
            boosting_type = self.space['boosting_type'],
            early_stopping_round = 20
            )
        lgb.fit(self.X_train, self.y_train,
                eval_set = [(self.X_valid, self.y_valid)]
            )
        
        lgb_valid_pred = lgb.predict_proba(self.X_valid)[:, 1]
        auc = roc_auc_score(self.y_valid, lgb_valid_pred)
        return -1.0 * auc   
    
    def hyper_param(self):
        trials = Trials()
        lgb_best = fmin(
            self.lgb_objective,
            space = self.space,
            algo=tpe.suggest,
            max_evals=self.iter,
            trials=trials,
            # 試行の過程を出力
            verbose=-1,
            early_stop_fn=no_progress_loss(300),
            show_progressbar = True,
        )
        return lgb_best
    
class Xgb_hyperopt(Base_model):
    def __init__(self, X_train, y_train, X_valid, y_valid, params, n_iter, random_state = 0):
        super().__init__(X_train, y_train, X_valid, y_valid, params, n_iter, random_state)

    def xgb_objective(self, args):
        xgb = XGBClassifier(
            max_depth = args['max_depth'],
            n_estimators = args['n_estimators'],
            learning_rate = args['learning_rate'],
            min_child_weight = args['min_child_weight'],
            reg_alpha = args['reg_alpha'],
            subsample = args['subsample'],
            colsample_bytree = args['colsample_bytree'],
            colsample_bylevel = args['colsample_bylevel'],
            random_state = self.random_state,
            n_jobs = -1,
            verbosity = 0,
            early_stopping_round = 20
            )
        xgb.fit(self.X_train, self.y_train,
                eval_set = [(self.X_valid, self.y_valid)]
            )
        
        xgb_valid_pred = xgb.predict_proba(self.X_valid)[:, 1]
        auc = roc_auc_score(self.y_valid, xgb_valid_pred)
        return -1.0 * auc   
    
    def hyper_param(self):
        trials = Trials()
        xgb_best = fmin(
            self.xgb_objective,
            space = self.space,
            algo=tpe.suggest,
            max_evals=self.iter,
            trials=trials,
            early_stop_fn=None,
            show_progressbar = True,
        )
        return xgb_best

class Cat_hyperopt(Base_model):
    def __init__(self, X_train, y_train, X_valid, y_valid, params, n_iter, random_state = 0):
        super().__init__(X_train, y_train, X_valid, y_valid, params, n_iter, random_state)

    def cat_objective(self, args):
        cat = CatBoostClassifier(
            num_boost_round = args['num_boost_round'],
            learning_rate = args['learning_rate'],
            depth = args['depth'],
            random_state = self.random_state,
            verbose = 0,
            early_stopping_rounds = 50
            )
        cat.fit(self.X_train, self.y_train,
                eval_set = [(self.X_valid, self.y_valid)]
            )
        
        cat_valid_pred = cat.predict_proba(self.X_valid)[:, 1]
        auc = roc_auc_score(self.y_valid, cat_valid_pred)
        return -1.0 * auc   
    
    def hyper_param(self):
        trials = Trials()
        cat_best = fmin(
            self.cat_objective,
            space = self.space,
            algo=tpe.suggest,
            max_evals=self.iter,
            trials=trials,
            early_stop_fn=no_progress_loss(100),
            show_progressbar = True,
        )
        return cat_best