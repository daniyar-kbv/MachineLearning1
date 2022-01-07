import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from typing import List, Optional


class RegressionFeatureSelector:
    __methods: List[str] = ['pearson', 'mutual_info', 'rfe', 'lin-reg', 'rf', 'lgbm']
    __n_jobs: int

    X_: pd.DataFrame
    X_scaled_: pd.DataFrame
    y_: pd.DataFrame
    feature_names: List[str]
    feature_support_: pd.DataFrame
    best_features_: List[str]

    def __init__(self,
                 methods='__all__',
                 n_jobs: Optional[int] = None):
        self.__n_jobs = n_jobs
        if methods != '__all__' \
                and isinstance(methods, List) \
                and all(isinstance(m, str) for m in methods):
            self.__methods = methods


    def __preprocess_dataset(self,
                             data_frame: pd.DataFrame,
                             numeric_features: List[str],
                             categorical_features: List[str],
                             target_name: str) -> (pd.DataFrame, np.ndarray, pd.DataFrame):
        df = data_frame
        df = df[numeric_features + categorical_features]
        train_df = pd.concat([df[numeric_features], pd.get_dummies(df[categorical_features])],
                             axis=1)
        features = train_df.columns
        train_df = pd.DataFrame(train_df, columns=features)
        X = train_df.copy()
        y = train_df[target_name]
        del X[target_name]
        X_scaled = MinMaxScaler().fit_transform(X)
        return X, X_scaled, y

    def __cor_selector(self, 
                       X: pd.DataFrame, 
                       y: pd.DataFrame, 
                       number_of_features: int) -> List[bool]:
        feature_names = X.columns.to_list()
        coefficients = [np.corrcoef(X[name], y)[0, 1] for name in feature_names]
        coefficients = [0 if np.isnan(coef) else coef for coef in coefficients]
        feature_indexes = np.argsort(np.abs(coefficients))[-number_of_features:]
        support = [index in feature_indexes for index, name in enumerate(feature_names)]
        return support

    def __mutual_info_regression_selector(self,
                               X: pd.DataFrame,
                               y: pd.DataFrame,
                               number_of_features: int) -> List[bool]:
        selector = SelectKBest(score_func=mutual_info_regression,
                               k=number_of_features)
        selector = selector.fit(X, y)
        return selector.get_support()

    def __rfe_selector(self,
                       X: np.ndarray,
                       y: pd.DataFrame,
                       number_of_features: int):
        model = RandomForestRegressor(n_estimators=50,
                                      n_jobs=self.__n_jobs,
                                      random_state=42)
        selector = RFE(estimator=model,
                       n_features_to_select=number_of_features,
                       step=1)
        selector = selector.fit(X, y)
        return selector.get_support()

    def __embedded_log_reg_selector(self,
                                    X: np.ndarray,
                                    y: pd.DataFrame,
                                    number_of_features: int):
        model = LinearRegression(normalize=True,
                                 n_jobs=self.__n_jobs)
        selector = SelectFromModel(model,
                                   max_features=number_of_features)
        selector = selector.fit(X, y)
        return selector.get_support()

    def __embedded_rf_selector(self,
                               X: np.ndarray,
                               y: pd.DataFrame,
                               number_of_features: int):
        model = RandomForestRegressor(n_estimators=50,
                                      n_jobs=self.__n_jobs,
                                      random_state=42)
        selector = SelectFromModel(model,
                                   max_features=number_of_features)
        embedded_selector = selector.fit(X, y)
        return embedded_selector.get_support()

    def __embedded_lgbm_selector(self,
                                 X: np.ndarray,
                                 y: pd.DataFrame,
                                 number_of_features: int):
        model = LGBMRegressor(n_estimators=500,
                              learning_rate=0.05,
                              num_leaves=32,
                              colsample_bytree=0.2,
                              reg_alpha=3,
                              reg_lambda=1,
                              min_split_gain=0.01,
                              min_child_weight=40,
                              n_jobs=self.__n_jobs,
                              random_state=42)
        selector = SelectFromModel(model,
                                   max_features=number_of_features)
        selector = selector.fit(X, y)
        return selector.get_support()

    def select_features(self,
                        data_frame: pd.DataFrame,
                        number_of_features: int,
                        target_name: str,
                        numeric_features: List[str] = [],
                        categorical_features: List[str] = []):
        X, X_scaled, y = self.__preprocess_dataset(data_frame=data_frame,
                                                   numeric_features=numeric_features,
                                                   categorical_features=categorical_features,
                                                   target_name=target_name)
        feature_names = X.columns.to_list()
        methods_support = {'Feature': feature_names}

        for method in self.__methods:
            print(f'Calculating {method}')
            if method == 'pearson' or self.__methods == '__all__':
                methods_support[method] = self.__cor_selector(X, y, number_of_features)
            if method == 'mutual_info' or self.__methods == '__all__':
                methods_support[method] = self.__mutual_info_regression_selector(X_scaled, y, number_of_features)
            if method == 'rfe' or self.__methods == '__all__':
                methods_support[method] = self.__rfe_selector(X_scaled, y, number_of_features)
            if method == 'lin-reg' or self.__methods == '__all__':
                methods_support[method] = self.__embedded_log_reg_selector(X_scaled, y, number_of_features)
            if method == 'rf' or self.__methods == '__all__':
                methods_support[method] = self.__embedded_rf_selector(X_scaled, y, number_of_features)
            if method == 'lgbm' or self.__methods == '__all__':
                methods_support[method] = self.__embedded_lgbm_selector(X_scaled, y, number_of_features)

        pd.set_option('display.max_rows', None)

        feature_selection_df = pd.DataFrame(methods_support)
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'],
                                                                ascending=False)

        feature_selection_df.index = range(1, len(feature_selection_df)+1)
        self.X_ = pd.DataFrame(X, columns=feature_names)
        self.X_scaled_ = pd.DataFrame(X_scaled, columns=feature_names)
        self.y_ = pd.DataFrame(y, columns=[target_name])
        self.feature_support_ = feature_selection_df
        self.best_features_ = feature_selection_df.head(number_of_features)['Feature'].tolist()
