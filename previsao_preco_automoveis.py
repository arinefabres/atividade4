# Importando bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados fictícios
data = {
    'Combustível': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol'],
    'Idade': [5, 3, 8, 2, 6, 7],
    'Quilometragem': [50000, 30000, 80000, 20000, 60000, 70000],
    'Preço': [30000, 35000, 25000, 40000, 33000, 28000]
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Definindo as variáveis independentes e a variável dependente (Preço)
X = df[['Combustível', 'Idade', 'Quilometragem']]
y = df['Preço']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo as colunas categóricas e numéricas
categorical_cols = ['Combustível']
numerical_cols = ['Idade', 'Quilometragem']

# Criando o ColumnTransformer para aplicar as transformações
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

# Criando o pipeline com o preprocessor e o modelo de regressão linear
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Calculando o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse}')
