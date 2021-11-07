from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from typing import List, Optional
import pandas as pd
import numpy as np


class ClassificationFeatureSelector:
    __methods: List[str] = ['pearson', 'mutual_info', 'rfe', 'lin-reg', 'rf', 'lgbm']
    __n_jobs: int

    feature_names: List[str]
    feature_support_: pd.DataFrame
    sorted_features_: List[str]

    def __init__(self,
                 methods='__all__',
                 n_jobs: Optional[int] = None):
        self.__n_jobs = n_jobs
        if methods != '__all__' \
                and isinstance(methods, List) \
                and all(isinstance(m, str) for m in methods):
            self.__methods = methods

    def __cor_selector(self,
                       X: pd.DataFrame,
                       y: pd.DataFrame,
                       number_of_features: int) -> List[bool]:
        feature_names = X.columns.to_list()
        coefficients = [np.corrcoef(pd.DataFrame(X[name], columns=[name]), y)[0, 1] for name in feature_names]
        coefficients = [0 if np.isnan(coef) else coef for coef in coefficients]
        feature_indexes = np.argsort(np.abs(coefficients))[-number_of_features:]
        support = [index in feature_indexes for index, name in enumerate(feature_names)]
        return support

    def __chi2_selector(self,
                               X: pd.DataFrame,
                               y: pd.DataFrame,
                               number_of_features: int) -> List[bool]:
        selector = SelectKBest(score_func=chi2,
                               k=number_of_features)
        selector = selector.fit(X, y)
        return selector.get_support()

    def __rfe_selector(self,
                       X: np.ndarray,
                       y: pd.DataFrame,
                       number_of_features: int):
        model = LogisticRegression()
        selector = RFE(estimator=model,
                       n_features_to_select=number_of_features,
                       step=1,
                       verbose=5)
        selector = selector.fit(X, y)
        return selector.get_support()

    def __embedded_log_reg_selector(self,
                                    X: np.ndarray,
                                    y: pd.DataFrame,
                                    number_of_features: int):
        model = LogisticRegression(n_jobs=self.__n_jobs)
        selector = SelectFromModel(model,
                                   max_features=number_of_features)
        selector = selector.fit(X, y)
        return selector.get_support()

    def __embedded_rf_selector(self,
                               X: np.ndarray,
                               y: pd.DataFrame,
                               number_of_features: int):
        model = RandomForestClassifier(n_estimators=50,
                                       n_jobs=self.__n_jobs,
                                       random_state=42,
                                       max_features=number_of_features)
        selector = SelectFromModel(model,
                                   max_features=number_of_features)
        embedded_selector = selector.fit(X, y)
        return embedded_selector.get_support()

    def __embedded_lgbm_selector(self,
                                 X: np.ndarray,
                                 y: pd.DataFrame,
                                 number_of_features: int):
        model = LGBMClassifier(n_estimators=500,
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

    def sort_features(self,
                      X: pd.DataFrame,
                      y: pd.DataFrame,
                      number_of_features: int):
        feature_names = X.columns.to_list()
        methods_support = {'Feature': feature_names}

        for method in self.__methods:
            print(f'Calculating {method}')
            if method == 'pearson' or self.__methods == '__all__':
                methods_support[method] = self.__cor_selector(X, y, number_of_features)
            if method == 'mutual_info' or self.__methods == '__all__':
                methods_support[method] = self.__chi2_selector(X, y, number_of_features)
            if method == 'rfe' or self.__methods == '__all__':
                methods_support[method] = self.__rfe_selector(X.to_numpy(), y, number_of_features)
            if method == 'lin-reg' or self.__methods == '__all__':
                methods_support[method] = self.__embedded_log_reg_selector(X.to_numpy(), y, number_of_features)
            if method == 'rf' or self.__methods == '__all__':
                methods_support[method] = self.__embedded_rf_selector(X.to_numpy(), y, number_of_features)
            if method == 'lgbm' or self.__methods == '__all__':
                methods_support[method] = self.__embedded_lgbm_selector(X.to_numpy(), y, number_of_features)

        pd.set_option('display.max_rows', None)

        feature_selection_df = pd.DataFrame(methods_support)
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'],
                                                                ascending=False)

        feature_selection_df.index = range(1, len(feature_selection_df)+1)
        self.feature_support_ = feature_selection_df
        self.sorted_features_ = feature_selection_df['Feature'].tolist()


df = pd.read_csv('https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv')

X = df.iloc[:,:-1]
y = pd.DataFrame(df.iloc[:,-1])

# Data cleaning
X.drop('user_id', inplace=True, axis=1)

# Simple imputing
mean_imputer = SimpleImputer(strategy='mean')
workclass_imputer = SimpleImputer(strategy='constant', fill_value='no_workclass')
occupation_imputer = SimpleImputer(strategy='constant', fill_value='no_occupation')
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)

X['age'] = mean_imputer.fit_transform(X['age'].to_numpy().reshape(-1, 1))
X['workclass'] = workclass_imputer.fit_transform(X['workclass'].to_numpy().reshape(-1, 1))
X['occupation'] = occupation_imputer.fit_transform(X['occupation'].to_numpy().reshape(-1, 1))

# Categorical features encoding
categorical_features = ['workclass', 'marital-status', 'occupation', 'race', 'sex']
for feature in categorical_features:
    X = pd.concat([X, pd.get_dummies(X[feature])], axis=1)
    X.drop(feature, inplace=True, axis=1)

# Iterative Imputing
X = pd.DataFrame(iterative_imputer.fit_transform(X), columns=X.columns)

# Normalization
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# Feature selection
selector = ClassificationFeatureSelector(n_jobs=-1)
selector.sort_features(X, y, number_of_features=10)
X = selector.sorted_features_[:10]

# Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rfc = RandomForestClassifier(n_estimators=100,
                             n_jobs=-1)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_accuracy = accuracy_score(y_test, rfc_pred)
print(f'RandomForestClassifier accuracy: {rfc_accuracy}')

svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f'SVC accuracy: {svm_accuracy}')

lr = LogisticRegression(random_state=42,
                        n_jobs=-1)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f'LogisticRegression accuracy: {lr_accuracy}')

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print(f'GaussianNB accuracy: {nb_accuracy}')

lgbm = LGBMClassifier(random_state=42,
                      n_jobs=-1)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_pred)
print(f'LGBMClassifier accuracy: {lgbm_accuracy}')

# Bagging
ys = pd.concat([pd.DataFrame(rfc_pred, columns=['rfc']),
                pd.DataFrame(svm_pred, columns=['svm']),
                pd.DataFrame(lr_pred, columns=['lr']),
                pd.DataFrame(nb_pred, columns=['nb']),
                pd.DataFrame(lr_pred, columns=['lr'])],
               axis=1)

ys['bagged'] = np.median(ys, axis=1)
bagged_pred = ys['bagged']
bagged_accuracy = accuracy_score(y_test, bagged_pred)
print(f'Bagged accuracy: {bagged_accuracy}')

print('LGBM alone showed better results than bagging')
