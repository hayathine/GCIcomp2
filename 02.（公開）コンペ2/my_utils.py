import os
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from typing import List
import pickle

from hyperopt import hp

class MyUtils:
    def __init__(self, input_path, output_path, exel_path):
        self.input_path = input_path
        self.output_path = output_path
        self.exel_path = exel_path
        self.train = None
        self.test = None
    
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
        self.train = train
        self.test = test
        return train, test, X_train, y_train, sample_sub
    
    def split_data(self, train):
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
        # 訓練データと評価データに分割
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        return X_train, X_valid, y_train, y_valid, X, y
    
    # pickleファイルにListを保存
    def save_pickle(self, data: List, file_name):
        with open(self.output_path + file_name + '.pickle', 'wb') as f:
            pickle.dump(data, f)

    
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

    # 欠損値のあるカラムの確認
    def missing_columns(self,):
        missing_columns = self.train.columns[self.train.isnull().sum() > 0]
        return missing_columns

    # 欠損値の詳細確認
    def missing_data(self, data):
        total = data.isnull().sum().sort_values(ascending = False)
        percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
        return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    def get_time():
        return datetime.datetime.now().strftime('%Y%m%d%H%M')
    
    # 複数のモデルで予測を行い、精度の違いを確認する関数
    def multi_model_predict(self, X_train, y_train, X_valid, y_valid, has_nan=False):
        
        nan_useable_models = [
                ('lgbm', lgb.LGBMClassifier(
                    verbose=-1
                    )),
                ('xgb', xgb.XGBClassifier(

                )),
                ('cat', cat.CatBoostClassifier(
                    verbose=0,
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=2,
                    random_state=0,
                    min_samples_split=5,
                    
                    verbose=0


                )),
            ]
        models = [
            
            ('gbc', GradientBoostingClassifier(

            )),  
            ('lor', LogisticRegression()),
            # ('lr', LinearRegression()),
            ('knn', KNeighborsClassifier()),
            # ('svc', SVC(probability=True)), # サポートベクターマシンは時間がかかるためコメントアウト
            ('nb', GaussianNB()),
            ('ada', AdaBoostClassifier()),
            ('dt', DecisionTreeClassifier()),
        ]
        
        # 欠損値がなければmodelsでも学習を行う
        if has_nan == False:


            for name, model in models:
                print(f'{model}_start')
                model.fit(X_train, y_train)
                y_pred_train = model.predict_proba(X_train)[:, 1]
                y_pred_valid = model.predict_proba(X_valid)[:, 1]
                print(f'{name} Train Score: {roc_auc_score(y_train, y_pred_train)}')
                print(f'{name} Valid Score: {roc_auc_score(y_valid, y_pred_valid)}')
        # 欠損値があれば欠損値を中央値で埋める
        else:
            # numpyに対してfillnaを行うとエラーが出るため、pandasに変換
            X_train = pd.DataFrame(X_train)
            X_valid = pd.DataFrame(X_valid)
            X_train.fillna(X_train.mean(), inplace=True)
            X_valid.fillna(X_valid.mean(), inplace=True)  
                
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
            for name, model in nan_useable_models:
                print(f'{model}_start')
                model.fit(X_train, y_train)
                y_pred_train = model.predict_proba(X_train)[:, 1]
                y_pred_valid = model.predict_proba(X_valid)[:, 1]
                print(f'{name} Train Score: {roc_auc_score(y_train, y_pred_train)}')
                print(f'{name} Valid Score: {roc_auc_score(y_valid, y_pred_valid)}')
        
    # 外れ値を除去
    def remove_outliers(self, train, columns, quantile=0.96):
        for column in columns:
            if train[column].max() <= 0:
                train.drop(train[train[column]  < train[column].quantile(1 - quantile)],axis=1)
                print(f"{column}_{train[column].shape}")
                print(f"外れ値：{train[column][train[column] < train[column].quantile(1 - quantile)]}")
            else:
                train.drop(train[train[column]  > train[column].quantile(quantile)],axis=1)
                print(f"{column}_{train[column].shape}")
                print(f"外れ値：{train[column][train[column] >= train[column].quantile(quantile)]}")
        return train

    # 標準化
    def standard_scaler(self, X_train, X_valid, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        return X_train, X_valid, X_test
        
    # カテゴリカルデータをラベルカウントエンコーディング Holdout Target Encoding https://www.codexa.net/target_encoding/
    def label_count_encoding(self, train, test, columns):
        for column in columns:
            train[column].fillna('missing', inplace=True)
            train[column] = train[column].map(train[column].value_counts().rank(ascending=False, method='first')).astype(int)
            test[column].fillna('missing', inplace=True)
            test[column] = test[column].map((test[column].value_counts().rank(ascending=False, method='first')).astype(int))
        return train, test
    
    # カテゴリカルデータをターゲットエンコーディング Holdout Target Encoding https://www.codexa.net/target_encoding/
    def target_encoding(self, X_train, X_test,  columns, random_state=0):
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state) # type: ignore
        for column in columns:
            x_box = np.zeros(len(X_train))
            x_box[:] = np.nan
            for train_idx, valid_idx in kf.split(X_train):
                train = X_train[[column, "TARGET"]].iloc[train_idx]
                valid = X_train[column].iloc[valid_idx]
                mean = train.groupby(column)["TARGET"].mean().to_frame()
                for i , m in mean.iterrows(): # i = name, m = value
                    # print(f"name {i}, itmes{m}")
                    # print(m.values)
                    for v in valid.index:
                        if valid[v] == i: 
                            x_box[v] = m.values
                    
            X_train[column] = x_box
            # Yも同様に埋める
            y_box = np.zeros(len(X_test))
            y_box[:] = np.nan
            maen = X_train.groupby(column)["TARGET"].mean().to_frame()
            for i , m in mean.iterrows():
                for v in range(len(X_test)):
                    if X_test[column][v] == i:
                        y_box[v] = m.values   
            X_test[column] = y_box
        return X_train ,X_test
    
 