import json

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def process_addons(data, num_top_addons=20):
    # Lista tuturor dotărilor din setul de date
    all_addons = pd.Series([addon for sublist in data['addons'] for addon in sublist])

    # Cele mai frecvente dotări
    top_addons = all_addons.value_counts().head(num_top_addons).index

    # Encodăm fiecare dotare semnificativă ca o caracteristică binară
    for addon in top_addons:
        data[f'addon_{addon}'] = data['addons'].apply(lambda x: addon in x)

    return data

data = pd.read_json('train.json')
test_data = pd.read_json('test.json')

data = process_addons(data)
test_data = process_addons(test_data)
numar_intrari = len(data)
print("Numărul total de intrări în setul de date:", numar_intrari)

train_data = data

# test_data = data.iloc[19000:21000]

X_train = train_data.drop(['pret', 'id'], axis=1)
y_train = train_data['pret']
X_test = test_data.drop(['id'], axis=1)

# X_test = test_data.drop(['pret', 'id'], axis=1)
# y_test = test_data['pret']

numeric_features = ['an', 'km', 'putere', 'capacitate_cilindrica']
categorical_features = ['marca', 'model', 'cutie_de_viteze', 'combustibil', 'transmisie', 'caroserie', 'culoare', 'optiuni_culoare']

numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

preprocessor = ColumnTransformer(
transformers=[
('num', Pipeline(steps=[('imputer', numeric_imputer), ('scaler', StandardScaler())]), numeric_features),
('cat', Pipeline(steps=[('imputer', categorical_imputer), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, gamma=0, reg_lambda=1, reg_alpha=0))
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

print(f"CV R2 scores: {cv_scores}")
print(f"Mean CV R2 score: {np.mean(cv_scores)}")

y_train_log = np.log(y_train)
# y_test_log = np.log(y_test)

model.fit(X_train, y_train_log)

predicted_prices_log = model.predict(X_test)
predicted_prices = np.exp(predicted_prices_log)

# predicted_prices_log = model.predict(X_test)
# predicted_prices = np.exp(predicted_prices_log)
# r2 = r2_score(y_test, predicted_prices)
# print(f"Scorul R2 pe setul de test este: {r2}")
# predicted_prices_rounded = np.round(np.exp(predicted_prices))

predicted_results = pd.DataFrame({'ID': test_data['id'], 'Pret prezis': np.round(predicted_prices)})

# results = pd.DataFrame({'ID': test_data['id'], 'Pret prezis': np.round(predicted_prices), 'Pret real': y_test})
print(predicted_results)

predicted_prices_rounded = np.round(predicted_prices).astype(int)

for i, (index, row) in enumerate(test_data.iterrows()):
    test_data.at[index, 'pret'] = predicted_prices_rounded[i]

test_data_dict = test_data.to_dict(orient='records')

with open('test.json', 'w', encoding='utf-8') as file:
    json.dump(test_data_dict, file, ensure_ascii=False, indent=4, separators=(',', ': '))

with open('test.json', 'r', encoding='utf-8') as file:
    updated_test_data = json.load(file)
print(updated_test_data[:5])
#
# plt.figure(figsize=(12, 6))
# sns.histplot(data=results, x='Pret real', color='blue', label='Pret real', kde=True, alpha=0.5)
# sns.histplot(data=results, x='Pret prezis', color='red', label='Pret prezis', kde=True, alpha=0.5)
# plt.title('Distribuția prețurilor reale vs. prețurilor prezise')
# plt.xlabel('Preț')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(12, 6))
# sns.scatterplot(data=results, x='Pret real', y='Pret prezis', alpha=0.5)
# plt.plot([results['Pret real'].min(), results['Pret real'].max()], [results['Pret real'].min(), results['Pret real'].max()], 'r--')
# plt.title('Relația dintre prețurile reale și prețurile prezise')
# plt.xlabel('Preț real')
# plt.ylabel('Preț prezis')
# plt.show()
#
# results['Loss'] = results['Pret prezis'] - results['Pret real']
# plt.figure(figsize=(12, 6))
# sns.histplot(data=results, x='Loss', kde=True, color='green')
# plt.title('Distribuția loss-ului')
# plt.xlabel('Loss')
# plt.show()
#
