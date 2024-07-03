from hyperopt import hp

class Params:
    lgb = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': hp.randint('num_leaves',2, 100),
    'max_depth': hp.randint('max_depth',2, 3),
    'n_estimators': hp.randint('n_estimators',800, 1200),
    'learning_rate': hp.uniform('learning_rate', 0.005 ,0.2),
    # 'min_child_samples': hp.randint('min_child_samples',5, 10),
    'reg_lamb': hp.uniform('reg_lamb', 0, 40),
    'bagging_freq': hp.randint('bagging_freq', 1, 10),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.8, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.1, 1),
    'min_data_in_leaf': hp.randint('min_data_in_leaf', 10, 100),
    'sub_sample': hp.uniform('sub_sample', 0.1, 1),
    }

    # https://qiita.com/c60evaporator/items/a9a049c3469f6b4872c6
    xgb = {
        'subsample': hp.uniform('subsample', 0.0, 1),
        'n_estimators': hp.randint('n_estimators', 800, 1200),
        'max_depth': hp.randint('max_depth', 2, 3),
        'learning_rate': hp.uniform('learning_rate', 0.005, 0.2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0, 40),
        'gamma': hp.uniform('gamma', 0, 40),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
    }

    lr = {
    'n_jobs': -1,
    'max_iter': hp.randint('max_iter', 100, 1000),
    'solver': hp.choice('solver', [ 'lbfgs', 'sag', 'saga']),
    'C': hp.uniform('C', 0.1, 1),
    'penalty': hp.choice('penalty', ['l2']),
    }