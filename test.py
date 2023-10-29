import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

df = pd.read_csv('train.csv')
# print(df.head(5))
selected_features = df[["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]]

# Создание dummy-переменных для столбцов Name, Sex, Ticket, Embarked
dummy_variables = pd.get_dummies(selected_features["Name"], prefix="name0")
selected_features = selected_features.drop("Name", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Sex"], prefix="sex0")
selected_features = selected_features.drop("Sex", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Ticket"], prefix="ticket0")
selected_features = selected_features.drop("Ticket", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Embarked"], prefix="embarked0")
selected_features = selected_features.drop("Embarked", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

dummy_variables = pd.get_dummies(selected_features["Cabin"], prefix="cabin0")
selected_features = selected_features.drop("Cabin", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)

selected_features = selected_features.astype(float)
selected_features = selected_features.dropna()
# print(selected_features.head(5), "\n")

features = selected_features.drop("Survived", axis=1)
target_var = selected_features["Survived"]

# Разделение данных на обучающий, валидационный и тестовый наборы
# train_data, remaining_data = train_test_split(selected_features, test_size=0.3)
# train_data_tar = train_data["Survived"]
# # print(train_data.head(5))
# train_data = train_data.drop("Survived", axis=1)
#
# test_data, validation_data = train_test_split(remaining_data, test_size=0.5)
# test_data_tar = test_data["Survived"]
# test_data = test_data.drop("Survived", axis=1)
#
# validation_data_tar = validation_data["Survived"]
# validation_data = validation_data.drop("Survived", axis=1)

features_train_data, features_test_data, target_train_data, target_test_data = train_test_split(features, target_var, test_size=0.2, random_state=43)

# Модель Random Forest
rf_model = RandomForestClassifier(random_state=43)
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
rf_grid = GridSearchCV(rf_model, rf_params, cv=5)
rf_grid.fit(features_train_data, target_train_data)
rf_best_model = rf_grid.best_estimator_

# Модель XGBoost
xgb_model = XGBClassifier(random_state=43)
xgb_params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5)
xgb_grid.fit(features_train_data, target_train_data)
xgb_best_model = xgb_grid.best_estimator_

# Модель Logistic Regression
lr_model = LogisticRegression(random_state=43)
lr_params = {'C': [0.1, 1, 10]}
lr_grid = GridSearchCV(lr_model, lr_params, cv=5)
lr_grid.fit(features_train_data, target_train_data)
lr_best_model = lr_grid.best_estimator_

# Модель KNN
knn_model = KNeighborsClassifier()
knn_params = {'n_neighbors': [3, 5, 7]}
knn_grid = GridSearchCV(knn_model, knn_params, cv=5)
knn_grid.fit(features_train_data, target_train_data)
knn_best_model = knn_grid.best_estimator_

# Оценка точности моделей на тестовой части
rf_accuracy = accuracy_score(target_test_data, rf_best_model.predict(features_test_data))
xgb_accuracy = accuracy_score(target_test_data, xgb_best_model.predict(features_test_data))
lr_accuracy = accuracy_score(target_test_data, lr_best_model.predict(features_test_data))
knn_accuracy = accuracy_score(target_test_data, knn_best_model.predict(features_test_data))

# Вывод точности моделей
print("Random Forest accuracy:", rf_accuracy)
print("XGBoost accuracy:", xgb_accuracy)
print("Logistic Regression accuracy:", lr_accuracy)
print("KNN accuracy:", knn_accuracy)

# Выбор 8 самых важных признаков с помощью Random Forest
rf_feature_selector = SelectKBest(score_func=f_classif, k=8)
rf_feature_selector.fit(features_train_data, target_train_data)
features_train_selected = rf_feature_selector.transform(features_train_data)
features_test_selected = rf_feature_selector.transform(features_test_data)

# Обучение модели Random Forest только на выбранных признаках
rf_model_selected = RandomForestClassifier(random_state=43)
rf_model_selected.fit(features_train_selected, target_train_data)

# Оценка точности модели Random Forest на выбранных признаках
rf_selected_accuracy = accuracy_score(target_test_data, rf_model_selected.predict(features_test_selected))
print("Random Forest on 8 features accuracy:", rf_selected_accuracy)

# Выбор 4 самых важных признаков с помощью Random Forest
rf_feature_selector = SelectKBest(score_func=f_classif, k=4)
rf_feature_selector.fit(features_train_data, target_train_data)
features_train_selected = rf_feature_selector.transform(features_train_data)
features_test_selected = rf_feature_selector.transform(features_test_data)

# Обучение модели Random Forest только на выбранных признаках
rf_model_selected = RandomForestClassifier(random_state=43)
rf_model_selected.fit(features_train_selected, target_train_data)

# Оценка точности модели Random Forest на выбранных признаках
rf_selected_accuracy = accuracy_score(target_test_data, rf_model_selected.predict(features_test_selected))
print("Random Forest on 4 features accuracy:", rf_selected_accuracy)

# Выбор 2 самых важных признаков с помощью Random Forest
rf_feature_selector = SelectKBest(score_func=f_classif, k=2)
rf_feature_selector.fit(features_train_data, target_train_data)
features_train_selected = rf_feature_selector.transform(features_train_data)
features_test_selected = rf_feature_selector.transform(features_test_data)

# Обучение модели Random Forest только на выбранных признаках
rf_model_selected = RandomForestClassifier(random_state=43)
rf_model_selected.fit(features_train_selected, target_train_data)

# Оценка точности модели Random Forest на выбранных признаках
rf_selected_accuracy = accuracy_score(target_test_data, rf_model_selected.predict(features_test_selected))
print("Random Forest on 2 features accuracy:", rf_selected_accuracy)
