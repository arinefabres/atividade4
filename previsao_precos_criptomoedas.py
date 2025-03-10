import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Função para obter dados históricos do Bitcoin
def get_bitcoin_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30'
    response = requests.get(url)
    data = response.json()
    
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

# Coletando os dados
df = get_bitcoin_data()

# Criando a variável 'target' para classificação
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)

# Removendo a última linha (porque não temos o próximo preço para ela).
df = df[:-1]

# Características (features) e alvo (target)
X = df[['price']]
y = df['target']

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Calculando a acurácia.
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')

# Exibindo a importância das variáveis.
print(f'Importância das variáveis: {model.feature_importances_}')
