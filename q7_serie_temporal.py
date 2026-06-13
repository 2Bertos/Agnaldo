import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

os.makedirs("resultados", exist_ok=True)
print("--- Iniciando Questão 7: Previsão de Série Temporal ---")

# 1. Carregar os dados
file_path = "Serie temporal.xlsx"
df = pd.read_excel(file_path, sheet_name="carga_eletrica")
# O arquivo tem a coluna 'PA(MW)' ou similar na primeira coluna
pa_series = df.iloc[:, 0].values.astype(float)

n_total = len(pa_series)
print(f"Total de amostras na série: {n_total}")

# 2. Construir o dataset com janelas deslizantes
# Entradas: PA(k-1), PA(k-2), PA(k-24)
# Saída: PA(k)
X = []
y = []

# O k deve começar em 24 (índice 24), pois precisamos de k-24
for k in range(24, n_total):
    pa_k_1 = pa_series[k-1]
    pa_k_2 = pa_series[k-2]
    pa_k_24 = pa_series[k-24]
    
    X.append([pa_k_1, pa_k_2, pa_k_24])
    y.append(pa_series[k])

X = np.array(X)
y = np.array(y)

# 3. Divisão Temporal dos Dados
# Treino: ~70%, Validação: ~15%, Teste-cego: ~15%
n_samples = len(X)
train_end = int(n_samples * 0.70)
val_end = int(n_samples * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

# Normalização usando MinMaxScaler (ajustado apenas no treino)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

X_test_scaled = scaler_X.transform(X_test)

# 4. Treinar a Rede PMC
# Usando o parâmetro de early stopping do scikit-learn (ele separa val set internamente,
# mas aqui juntamos train+val se quisermos ou deixamos ele fazer sozinho).
# Para controle total, usaremos apenas X_train e deixaremos ele tirar 15% para early_stopping
mlp = MLPRegressor(hidden_layer_sizes=(15,), 
                   activation='relu', 
                   solver='adam', 
                   max_iter=1000, 
                   early_stopping=True,
                   validation_fraction=0.15,
                   random_state=42)

print("Treinando previsor neural...")
mlp.fit(X_train_scaled, y_train_scaled)
print(f"Treinamento concluído em {mlp.n_iter_} épocas.")

# 5. Avaliação de 1 passo à frente no conjunto de teste
y_pred_1passo_scaled = mlp.predict(X_test_scaled)
y_pred_1passo = scaler_y.inverse_transform(y_pred_1passo_scaled.reshape(-1, 1)).ravel()

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mape

rmse_1, mape_1 = calculate_metrics(y_test, y_pred_1passo)
print(f"Teste Cego (1 passo à frente) -> RMSE: {rmse_1:.2f}, MAPE: {mape_1:.2f}%")

plt.figure(figsize=(12, 5))
plt.plot(y_test, label='Real')
plt.plot(y_pred_1passo, label='Previsto (1 passo)', alpha=0.8)
plt.title('Previsão: 1 Passo à Frente no Teste-Cego')
plt.xlabel('Hora')
plt.ylabel('Potência Ativa (MW)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("resultados", "q7_serie_1passo.png"))
plt.close()

# 6. Previsão Recursiva de 1 a 24 passos à frente
# Faremos a previsão recursiva começando do início do conjunto de teste.
# Para ter métricas confiáveis de N passos, precisaríamos rodar várias janelas de 24h.
# Aqui rodaremos em janelas contíguas de 24h ao longo do teste e calcularemos a média das métricas.

horizons = 24
rmses = []
mapes = []
all_preds_24 = []
all_reals_24 = []

# Necessitamos do histórico completo até val_end para pegar o k-24 corretamente
historico_real = list(pa_series[:24 + val_end]) 
pontos_inicio_teste = range(val_end, len(pa_series) - horizons, 24)

print("Realizando previsões recursivas de 1 a 24 passos...")

metrics_por_horizonte = {h: {'reals': [], 'preds': []} for h in range(1, horizons + 1)}

for start_idx in pontos_inicio_teste:
    # A previsão recursiva trabalha iterativamente
    # Histórico atual de previsões (inicia com os dados reais antes do start_idx)
    current_history = list(pa_series[:start_idx])
    
    preds = []
    reals = []
    
    for h in range(1, horizons + 1):
        k = start_idx + h - 1 # Índice absoluto que queremos prever
        
        # Obter os lags (podem ser reais ou já previstos se h > 1)
        pa_k_1 = current_history[-1]
        pa_k_2 = current_history[-2]
        pa_k_24 = current_history[-24]
        
        # Escalar a entrada
        x_in = np.array([[pa_k_1, pa_k_2, pa_k_24]])
        x_in_scaled = scaler_X.transform(x_in)
        
        # Prever
        y_out_scaled = mlp.predict(x_in_scaled)
        y_out = scaler_y.inverse_transform(y_out_scaled.reshape(-1, 1))[0][0]
        
        # Atualizar o histórico COM a previsão (comportamento recursivo)
        current_history.append(y_out)
        
        # Salvar para métricas
        preds.append(y_out)
        real_val = pa_series[k]
        reals.append(real_val)
        
        metrics_por_horizonte[h]['reals'].append(real_val)
        metrics_por_horizonte[h]['preds'].append(y_out)
        
    all_preds_24.extend(preds)
    all_reals_24.extend(reals)

# Calcular métricas por horizonte
horizontes_x = []
mape_y = []
rmse_y = []

for h in range(1, horizons + 1):
    r = np.array(metrics_por_horizonte[h]['reals'])
    p = np.array(metrics_por_horizonte[h]['preds'])
    rmse, mape = calculate_metrics(r, p)
    
    horizontes_x.append(h)
    rmse_y.append(rmse)
    mape_y.append(mape)

# Salvar métricas
df_metrics = pd.DataFrame({'Horizonte': horizontes_x, 'RMSE': rmse_y, 'MAPE(%)': mape_y})
df_metrics.to_csv(os.path.join("resultados", "q7_metricas_por_horizonte.csv"), index=False)
print("Métricas por horizonte salvas em resultados/q7_metricas_por_horizonte.csv")

# Plotar RMSE e MAPE
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(horizontes_x, rmse_y, 'g-', marker='o', label='RMSE')
ax2.plot(horizontes_x, mape_y, 'b-', marker='x', label='MAPE (%)')

ax1.set_xlabel('Horizonte de Previsão (passos/horas)')
ax1.set_ylabel('RMSE (MW)', color='g')
ax2.set_ylabel('MAPE (%)', color='b')
plt.title('Erro de Previsão Recursiva por Horizonte')
fig.tight_layout()
plt.grid(True)
plt.savefig(os.path.join("resultados", "q7_mape_rmse_horizonte.png"))
plt.close()

# Plot ilustrativo: série completa vs previsões recursivas de 24h concatenadas
plt.figure(figsize=(12, 5))
plt.plot(all_reals_24, label='Real')
plt.plot(all_preds_24, label='Previsto Recursivo (blocos de 24h)', alpha=0.8)
plt.title('Teste-cego: Previsão Recursiva a cada 24 horas')
plt.xlabel('Hora (no conjunto de teste)')
plt.ylabel('Potência Ativa (MW)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("resultados", "q7_serie_24passos.png"))
plt.close()

print("Gráficos de previsão recursiva gerados.")
print("--- Questão 7 Concluída ---")
