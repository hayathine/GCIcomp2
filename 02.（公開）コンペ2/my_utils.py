import os
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from hyperopt import hp

class MyUtils:
    params = {
    'random_state': 0,
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
    'bagging_fraction': hp.uniform('bagging_fraction', 0.1, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.1, 1),
    'min_data_in_leaf': hp.randint('min_data_in_leaf', 10, 100),
    }

    def __init__(self, input_path, output_path, exel_path):
        self.input_path = input_path
        self.output_path = output_path
        self.exel_path = exel_path
    
    def get_output_path(self):
        return self.output_path
    
    def get_input_path(self):
        return self.input_path
    


    # データの読み込み
    # INPUT_DIRにtrain.csvなどのデータを置いているディレクトリを指定してください。
    def load_data(self):
        train = pd.read_csv(self.input_path + "train.csv")
        test = pd.read_csv(self.input_path + "test.csv")
        sample_sub = pd.read_csv(self.input_path + "sample_submission.csv")
        X_train = train.drop('TARGET',axis=1)
        y_train = train['TARGET']
        return train, test, X_train, y_train, sample_sub
    
    def split_data(self, train, test):
        """_summary_
        目的変数と説明変数に分割
        Args:
            train (_type_): _description_
            test (_type_): _description_

        Returns:
            X, y: train.drop-"TARGET"
            X_train, X_valid, y_train, y_valid, X_test: train_test_split

        """
        X = train.drop("TARGET", axis=1).values
        y = train["TARGET"].values
        X_test = test.values
        # 訓練データと評価データに分割
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        return X_train, X_valid, y_train, y_valid, X_test, X, y
    
    # exel表にparameter,scoreを記録する関数
    def save_score_to_exel(self, exel_path , best: dict, score, time):
        columns = list(best.keys())
        columns.append('score')
        columns.append('time')
        old = pd.DataFrame(columns = columns)
        if not os.path.exists(exel_path):
            old.to_excel(exel_path)
        old = pd.read_excel(exel_path, index_col=0)
        data = list(best.values())
        data.append(score)
        data.append(time)
        new = np.array([data])
        score_data = pd.DataFrame(columns = columns, data = new)
        new = pd.concat([old, score_data])
        new.to_excel(exel_path)

    # 日数を週単位にbin分割
    def from_days_to_week_bin(self, data, columns):
        for column in columns:
            data[column+'_bin_week'] = np.floor(data[column]/7)
        return data

    # 日数を月単位にbin分割
    def from_days_to_month_bin(self, data, columns):
        for column in columns:
            data[column+'_bin_by_month'] = np.floor(data[column]/30)
        return data
    
    # 日数を年単位にbin分割
    def from_days_to_year_bin(self,data, columns):
        for column in columns:
            data[column+'_bin_by_year'] = np.floor(data[column]/365)
        return data

    # 欠損値の確認
    def missing_data(self, data):
        total = data.isnull().sum().sort_values(ascending = False)
        percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
        return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # ラベルカウントエンコーディング用の関数
    def label_count(self, X, Y, columns):
        for column in columns:
            if X[column].isnull().sum() > 0:
                X[column].fillna('missing', inplace=True)
            X[column] = X[column].map(X[column].value_counts().rank(ascending=False, method='first'))
            if Y[column].isnull().sum() > 0:
                Y[column].fillna('missing', inplace=True)
            Y[column] = Y[column].map(X[column].value_counts().rank(ascending=False, method='first'))
        return X, Y
    
    def get_time():
        return datetime.datetime.now().strftime('%Y%m%d%H%M')
    
    # 複数のモデルで予測を行い、精度の違いを確認する関数
    def multi_model_predict(self, X_train, y_train, X_valid, y_valid, hasNan):
        if hasNan==True:
            models = [
                ('gbc', GradientBoostingClassifier()),  
                ('lgbm', lgb.LGBMClassifier(verbose=-1)),
                ('xgb', xgb.XGBClassifier()),
                ('cat', cat.CatBoostClassifier()),
                ('rf', RandomForestClassifier()),

            ]
        else:
            models = [
                ('lr', LogisticRegression()),
                ('knn', KNeighborsClassifier()),
                ('svc', SVC(probability=True)),
                ('nb', GaussianNB()),
                ('ada', AdaBoostClassifier()),
                ('dt', DecisionTreeClassifier()),
            ]
            # 数値型の特徴量は標準化処理
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

        for name, model in models:
            print(f'{model}_start')
            model.fit(X_train, y_train)
            y_pred_train = model.predict_proba(X_train)[:, 1]
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            

            print(f'{name} Train Score: {roc_auc_score(y_train, y_pred_train)}')
            print(f'{name} Valid Score: {roc_auc_score(y_valid, y_pred_valid)}')

    # 外れ値の除去 標準偏差の3倍以上の値を除去
    def remove_outliers(self, data, columns):
        for column in columns:
            data = data[np.abs(data[column] - data[column].mean()) <= (3 * data[column].std())]
        return data

    # 標準化
    def standard_scaler(self, X_train, X_valid, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        return X_train, X_valid, X_test
        
    # 欠損値に対してターゲットエンコーディング use leave one out
    def target_encoding(self, X, Y, columns):
        for column in columns:
            target_mean = Y[column].mean()
            X[column] = X[column].map(target_mean)
            if X[column].isnull().sum() > 0:
                X[column].fillna(Y.mean(), inplace=True)
        return X, Y