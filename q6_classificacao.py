import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import os

os.makedirs("resultados", exist_ok=True)

print("--- Iniciando Questão 6: Classificação com PMC ---")

file_path = "Classificação.xlsx"

# 1. Carregar e parsear dados de treinamento
df_train_raw = pd.read_excel(file_path, sheet_name="tab_treinamento")

train_data = []
# O iterrows() pula o cabeçalho 'x1', 'x2', 'x3', 'classe' (se estiver na linha 0)
for _, row in df_train_raw.iterrows():
    # A primeira coluna contém a string concatenada
    val = row.iloc[0]
    if isinstance(val, str):
        # Converter vírgula para ponto e splitar por espaços
        val_str = val.replace(',', '.')
        parts = val_str.split()
        if len(parts) >= 4:
            train_data.append([float(p) for p in parts[:4]])

df_train = pd.DataFrame(train_data, columns=['x1', 'x2', 'x3', 'classe'])

X_train = df_train[['x1', 'x2', 'x3']].values
y_train = df_train['classe'].values

# O MLPClassifier do sklearn precisa que os labels de classe sejam -1 ou 1 (int)
y_train = np.where(y_train > 0, 1, -1)

print(f"Treino carregado. Shape X: {X_train.shape}, Shape y: {y_train.shape}")

# 2. Carregar e parsear dados de teste
df_test_raw = pd.read_excel(file_path, sheet_name="tab_teste")

test_data = []
for _, row in df_test_raw.iterrows():
    val = row.iloc[0]
    if isinstance(val, str):
        val_str = val.replace(',', '.')
        parts = val_str.split()
        if len(parts) >= 3:
            test_data.append([float(p) for p in parts[:3]])

X_test = np.array(test_data)
print(f"Teste carregado. Shape X_test: {X_test.shape}")

# 3. Função para treinar e extrair informações do modelo
def train_and_extract(seed, name):
    print(f"\n--- Treinamento: {name} (Random Seed: {seed}) ---")
    
    # Arquitetura simples para classificação: 1 camada oculta com 5 neurônios
    mlp = MLPClassifier(hidden_layer_sizes=(5,), 
                        activation='tanh', 
                        solver='lbfgs', # lbfgs é muito bom para datasets pequenos
                        max_iter=20, 
                        random_state=seed)
    
    mlp.fit(X_train, y_train)
    
    print(f"Número de épocas até convergência: {mlp.n_iter_}")
    
    # Extrair pesos e bias
    # mlp.coefs_ é uma lista de matrizes de pesos (camada_i para camada_i+1)
    # mlp.intercepts_ é uma lista de bias
    
    weights_summary = []
    
    # Camada Entrada -> Oculta
    w_in_hid = mlp.coefs_[0]
    b_hid = mlp.intercepts_[0]
    for i in range(w_in_hid.shape[0]):
        for j in range(w_in_hid.shape[1]):
            weights_summary.append({'Camada': 'Entrada->Oculta', 'De_Neuro': f'In_{i+1}', 'Para_Neuro': f'Hid_{j+1}', 'Peso/Bias': 'Peso', 'Valor': w_in_hid[i, j]})
    for j in range(len(b_hid)):
        weights_summary.append({'Camada': 'Entrada->Oculta', 'De_Neuro': 'Bias', 'Para_Neuro': f'Hid_{j+1}', 'Peso/Bias': 'Bias', 'Valor': b_hid[j]})
        
    # Camada Oculta -> Saída
    w_hid_out = mlp.coefs_[1]
    b_out = mlp.intercepts_[1]
    for i in range(w_hid_out.shape[0]):
        for j in range(w_hid_out.shape[1]):
            weights_summary.append({'Camada': 'Oculta->Saída', 'De_Neuro': f'Hid_{i+1}', 'Para_Neuro': f'Out_{j+1}', 'Peso/Bias': 'Peso', 'Valor': w_hid_out[i, j]})
    for j in range(len(b_out)):
        weights_summary.append({'Camada': 'Oculta->Saída', 'De_Neuro': 'Bias', 'Para_Neuro': f'Out_{j+1}', 'Peso/Bias': 'Bias', 'Valor': b_out[j]})
        
    df_weights = pd.DataFrame(weights_summary)
    path = os.path.join("resultados", f"q6_pesos_{name}.csv")
    df_weights.to_csv(path, index=False)
    print(f"Pesos salvos em: {path}")
    
    # Prever no teste
    y_pred = mlp.predict(X_test)
    return mlp, y_pred, df_weights

# Realizar dois treinamentos com seeds diferentes
mlp_t1, y_pred_t1, df_w_t1 = train_and_extract(seed=42, name="T1")
mlp_t2, y_pred_t2, df_w_t2 = train_and_extract(seed=123, name="T2")

# 4. Consolidar os resultados de classificação
df_results = pd.DataFrame({
    'Amostra': range(1, len(X_test) + 1),
    'x1': X_test[:, 0],
    'x2': X_test[:, 1],
    'x3': X_test[:, 2],
    'y (T1)': y_pred_t1,
    'y (T2)': y_pred_t2
})

res_path = os.path.join("resultados", "q6_classificacao_teste.csv")
df_results.to_csv(res_path, index=False)

print("\n--- Resultados da Classificação ---")
print(df_results.to_string(index=False))

# Gerar gráfico comparativo
plt.figure(figsize=(10, 5))
x_indices = np.arange(len(X_test)) + 1
width = 0.35

plt.bar(x_indices - width/2, y_pred_t1, width, label='Treinamento T1 (Seed 42)', color='royalblue', alpha=0.8)
plt.bar(x_indices + width/2, y_pred_t2, width, label='Treinamento T2 (Seed 123)', color='darkorange', alpha=0.8)

plt.xlabel('Amostra de Teste')
plt.ylabel('Classe Predita (-1 ou 1)')
plt.title('Comparação de Classificação (T1 vs T2) com Máx 20 Épocas')
plt.xticks(x_indices)
plt.yticks([-1, 1])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plot_path = os.path.join("resultados", "q6_comparacao_classes.png")
plt.savefig(plot_path)
plt.close()
print(f"\nGráfico comparativo salvo em: {plot_path}")

# Explicação exigida na questão:
expl_path = os.path.join("resultados", "q6_explicacao.txt")
with open(expl_path, "w", encoding="utf-8") as f:
    f.write("a) Executados dois treinamentos com sementes aleatórias diferentes (42 e 123).\n")
    f.write(f"b) Tabelas de pesos salvas em q6_pesos_T1.csv e q6_pesos_T2.csv.\n")
    f.write(f"   Épocas T1: {mlp_t1.n_iter_}\n")
    f.write(f"   Épocas T2: {mlp_t2.n_iter_}\n\n")
    f.write("   EXPLICAÇÃO: O número de épocas varia porque a inicialização aleatória dos pesos\n")
    f.write("   posiciona o algoritmo em diferentes pontos da superfície de erro (loss landscape).\n")
    f.write("   Como o algoritmo tenta descer o gradiente para encontrar o erro mínimo, diferentes pontos\n")
    f.write("   de partida resultam em trajetórias diferentes, levando mais ou menos tempo (épocas) para convergir.\n\n")
    f.write("   CRITÉRIO DE PARADA: O Scikit-learn (solver='lbfgs' ou 'adam') para quando a melhoria na função de perda\n")
    f.write("   (loss) ou na pontuação de validação for menor que 'tol' (padrão 1e-4) por um determinado\n")
    f.write("   número consecutivo de iterações ('n_iter_no_change'), indicando que a rede já aprendeu o máximo possível.\n\n")
    f.write("c) A classificação das 10 amostras está salva em q6_classificacao_teste.csv e apresentada na execução.\n")

print("\nExplicações salvas em resultados/q6_explicacao.txt")
print("--- Questão 6 Concluída ---")
