import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('titanic_prepared.csv')

selected_features = df.drop('label', axis=1)
target_variable = df['label']

features_train_data, features_test_data, target_train_data, target_test_data = train_test_split(selected_features, target_variable, test_size=0.1, random_state=48)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=43)
dt_model.fit(features_train_data, target_train_data)
dt_predictions = dt_model.predict(features_test_data)
dt_accuracy = accuracy_score(target_test_data, dt_predictions)

# XGBoost
xgb_model = XGBClassifier(random_state=43)
xgb_model.fit(features_train_data, target_train_data)
xgb_predictions = xgb_model.predict(features_test_data)
xgb_accuracy = accuracy_score(target_test_data, xgb_predictions)

# Logistic Regression
lr_model = LogisticRegression(random_state=43)
lr_model.fit(features_train_data, target_train_data)
lr_predictions = lr_model.predict(features_test_data)
lr_accuracy = accuracy_score(target_test_data, lr_predictions)

# Вывод точности моделей
print("Decision Tree accuracy:", dt_accuracy)
print("XGBoost accuracy:", xgb_accuracy)
print("Logistic Regression accuracy:", lr_accuracy)

# Выбор 2 самых важных признаков с помощью Decision Tree
dt_feature_selector = SelectKBest(score_func=f_classif, k=2)
dt_feature_selector.fit(features_train_data, target_train_data)
features_train_selected = dt_feature_selector.transform(features_train_data)
features_test_selected = dt_feature_selector.transform(features_test_data)

# Обучение модели Decision Tree только на выбранных признаках
dt_model_selected = DecisionTreeClassifier(random_state=43)
dt_model_selected.fit(features_train_selected, target_train_data)

# Оценка точности модели Decision Tree на выбранных признаках
dt_selected_accuracy = accuracy_score(target_test_data, dt_model_selected.predict(features_test_selected))
print("Decision Tree on 2 features accuracy:", dt_selected_accuracy)
