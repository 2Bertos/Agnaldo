"""
Questão 6 - Classificação de Óleo com PMC
=========================================
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# Configurações gerais
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Classificação.xlsx"
OUT_DIR = BASE_DIR / "resultados_q6"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Questão 6 - Classificação com PMC")
print("=" * 70)

# -------------------------------------------------------------------------
# 1. Funções auxiliares de leitura
# -------------------------------------------------------------------------
def extract_numbers_from_cell(value):
    """Extrai números de uma célula que pode estar em formato texto com vírgula decimal."""
    if not isinstance(value, str):
        return []
    text = value.replace(",", ".")
    # Aceita números em notação decimal e científica.
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return [float(n) for n in nums]


def load_training_data(file_path: Path):
    raw = pd.read_excel(file_path, sheet_name="tab_treinamento", header=None)
    rows = []
    for _, row in raw.iterrows():
        nums = extract_numbers_from_cell(row.iloc[0])
        if len(nums) >= 4:
            rows.append(nums[:4])

    if not rows:
        raise ValueError("Não foi possível ler os dados de treinamento.")

    df = pd.DataFrame(rows, columns=["x1", "x2", "x3", "d"])
    X = df[["x1", "x2", "x3"]].to_numpy(dtype=float)
    d = df["d"].to_numpy(dtype=float).reshape(-1, 1)
    d = np.where(d >= 0, 1.0, -1.0)
    return df, X, d


def load_test_data(file_path: Path):
    raw = pd.read_excel(file_path, sheet_name="tab_teste", header=None)
    rows = []
    for _, row in raw.iterrows():
        nums = extract_numbers_from_cell(row.iloc[0])
        if len(nums) >= 3:
            rows.append(nums[:3])

    if not rows:
        raise ValueError("Não foi possível ler os dados de teste.")

    X = np.array(rows, dtype=float)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    return df, X


# -------------------------------------------------------------------------
# 2. Classe da PMC manual
# -------------------------------------------------------------------------
class ManualMLPClassifier:
    """PMC simples: 3 entradas -> camada oculta tanh -> 1 saída tanh."""

    def __init__(
        self,
        n_inputs=3,
        n_hidden=5,
        init_min=0.0,
        init_max=1.0,
        learning_rate=0.05,
        max_epochs=100_000,
        mse_goal=1e-3,
        random_state=42,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.init_min = init_min
        self.init_max = init_max
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.mse_goal = mse_goal
        self.random_state = random_state

        rng = np.random.default_rng(random_state)
        self.W1 = rng.uniform(init_min, init_max, size=(n_inputs, n_hidden))
        self.b1 = rng.uniform(init_min, init_max, size=(1, n_hidden))
        self.W2 = rng.uniform(init_min, init_max, size=(n_hidden, 1))
        self.b2 = rng.uniform(init_min, init_max, size=(1, 1))

        # Guarda cópias para demonstrar que a inicialização respeitou o intervalo.
        self.W1_initial = self.W1.copy()
        self.b1_initial = self.b1.copy()
        self.W2_initial = self.W2.copy()
        self.b2_initial = self.b2.copy()

        self.loss_curve_ = []
        self.n_epochs_ = 0
        self.stop_reason_ = ""

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative_from_output(a):
        return 1.0 - a**2

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self.tanh(z2)
        return a1, y_hat

    def fit(self, X, y):
        n = X.shape[0]

        for epoch in range(1, self.max_epochs + 1):
            a1, y_hat = self.forward(X)
            error = y_hat - y
            mse = float(np.mean(error**2))
            self.loss_curve_.append(mse)

            if mse <= self.mse_goal:
                self.n_epochs_ = epoch
                self.stop_reason_ = f"Erro MSE atingiu a meta definida ({self.mse_goal:g})."
                break

            # Backpropagation para MSE com saída tanh.
            delta2 = (2.0 / n) * error * self.tanh_derivative_from_output(y_hat)
            dW2 = a1.T @ delta2
            db2 = np.sum(delta2, axis=0, keepdims=True)

            delta1 = (delta2 @ self.W2.T) * self.tanh_derivative_from_output(a1)
            dW1 = X.T @ delta1
            db1 = np.sum(delta1, axis=0, keepdims=True)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
        else:
            self.n_epochs_ = self.max_epochs
            self.stop_reason_ = "Treinamento parou por atingir o número máximo de épocas."

        return self

    def predict_continuous(self, X):
        _, y_hat = self.forward(X)
        return y_hat.ravel()

    def predict_class(self, X):
        y_hat = self.predict_continuous(X)
        return np.where(y_hat >= 0, 1, -1)

    def final_mse(self, X, y):
        y_hat = self.predict_continuous(X).reshape(-1, 1)
        return float(np.mean((y_hat - y) ** 2))

    def accuracy(self, X, y):
        pred = self.predict_class(X).reshape(-1, 1)
        return float(np.mean(pred == y))


# -------------------------------------------------------------------------
# 3. Funções para salvar pesos e bias
# -------------------------------------------------------------------------
def weights_to_dataframe(model: ManualMLPClassifier, name: str):
    rows = []

    # Entrada -> Oculta
    for i in range(model.W1.shape[0]):
        for j in range(model.W1.shape[1]):
            rows.append({
                "Treinamento": name,
                "Camada": "Entrada->Oculta",
                "Origem": f"x{i+1}",
                "Destino": f"h{j+1}",
                "Tipo": "Peso",
                "Valor_final": model.W1[i, j],
            })

    for j in range(model.b1.shape[1]):
        rows.append({
            "Treinamento": name,
            "Camada": "Entrada->Oculta",
            "Origem": "Bias",
            "Destino": f"h{j+1}",
            "Tipo": "Bias",
            "Valor_final": model.b1[0, j],
        })

    # Oculta -> Saída
    for j in range(model.W2.shape[0]):
        rows.append({
            "Treinamento": name,
            "Camada": "Oculta->Saída",
            "Origem": f"h{j+1}",
            "Destino": "y",
            "Tipo": "Peso",
            "Valor_final": model.W2[j, 0],
        })

    rows.append({
        "Treinamento": name,
        "Camada": "Oculta->Saída",
        "Origem": "Bias",
        "Destino": "y",
        "Tipo": "Bias",
        "Valor_final": model.b2[0, 0],
    })

    return pd.DataFrame(rows)


def initial_range_summary(model: ManualMLPClassifier):
    all_initial = np.concatenate([
        model.W1_initial.ravel(), model.b1_initial.ravel(),
        model.W2_initial.ravel(), model.b2_initial.ravel(),
    ])
    return float(all_initial.min()), float(all_initial.max())


# -------------------------------------------------------------------------
# 4. Carregar dados e padronizar entradas
# -------------------------------------------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")

train_df, X_train_raw, d_train = load_training_data(DATA_FILE)
test_df, X_test_raw = load_test_data(DATA_FILE)

print(f"Amostras de treinamento: {X_train_raw.shape[0]}")
print(f"Amostras de teste: {X_test_raw.shape[0]}")
print("Distribuição das classes no treino:")
print(pd.Series(d_train.ravel()).value_counts().sort_index().to_string())

# A padronização melhora o treinamento da PMC.
# Os pesos finais salvos pertencem à rede treinada com as entradas padronizadas.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Salvar os dados padronizados usados no treinamento, para rastreabilidade.
pd.DataFrame(X_train, columns=["x1_pad", "x2_pad", "x3_pad"]).assign(d=d_train.ravel()).to_csv(
    OUT_DIR / "q6_dados_treinamento_padronizados.csv", index=False
)

# -------------------------------------------------------------------------
# 5. Treinamentos exigidos no enunciado
# -------------------------------------------------------------------------
configs = [
    {
        "name": "T1",
        "intervalo": "[0, 1]",
        "init_max": 1.0,
        "seed": 42,
    },
    {
        "name": "T2",
        "intervalo": "[0, 2]",
        "init_max": 2.0,
        "seed": 43,
    },
]

models = {}
summary_rows = []
weight_tables = []

for cfg in configs:
    name = cfg["name"]
    print(f"\nTreinamento {name}: inicialização dos pesos e bias em {cfg['intervalo']}")

    model = ManualMLPClassifier(
        n_inputs=3,
        n_hidden=5,
        init_min=0.0,
        init_max=cfg["init_max"],
        learning_rate=0.05,
        max_epochs=100_000,
        mse_goal=1e-3,
        random_state=cfg["seed"],
    )
    model.fit(X_train, d_train)
    models[name] = model

    init_min_obtained, init_max_obtained = initial_range_summary(model)
    final_mse = model.final_mse(X_train, d_train)
    acc_train = model.accuracy(X_train, d_train)

    print(f"  Menor peso/bias inicial: {init_min_obtained:.6f}")
    print(f"  Maior peso/bias inicial: {init_max_obtained:.6f}")
    print(f"  Épocas executadas       : {model.n_epochs_}")
    print(f"  MSE final no treino     : {final_mse:.6f}")
    print(f"  Acurácia no treino      : {acc_train*100:.2f}%")
    print(f"  Critério de parada      : {model.stop_reason_}")

    summary_rows.append({
        "Treinamento": name,
        "Intervalo_inicial_pedido": cfg["intervalo"],
        "Menor_peso_bias_inicial": init_min_obtained,
        "Maior_peso_bias_inicial": init_max_obtained,
        "Neuronios_ocultos": model.n_hidden,
        "Taxa_aprendizado": model.learning_rate,
        "Meta_MSE": model.mse_goal,
        "Max_epocas": model.max_epochs,
        "Epocas_executadas": model.n_epochs_,
        "MSE_final_treino": final_mse,
        "Acuracia_treino": acc_train,
        "Criterio_parada": model.stop_reason_,
    })

    df_w = weights_to_dataframe(model, name)
    df_w.to_csv(OUT_DIR / f"q6_pesos_bias_{name}.csv", index=False)
    weight_tables.append(df_w)

pd.DataFrame(summary_rows).to_csv(OUT_DIR / "q6_resumo_treinamentos.csv", index=False)
pd.concat(weight_tables, ignore_index=True).to_csv(OUT_DIR / "q6_pesos_bias_T1_T2.csv", index=False)

# -------------------------------------------------------------------------
# 6. Classificação das novas amostras
# -------------------------------------------------------------------------
results = test_df.copy()
for name, model in models.items():
    y_cont = model.predict_continuous(X_test)
    y_class = model.predict_class(X_test)
    results[f"saida_continua_{name}"] = y_cont
    results[f"classe_{name}"] = y_class

results.insert(0, "Amostra", np.arange(1, len(results) + 1))
results.to_csv(OUT_DIR / "q6_classificacao_teste.csv", index=False)

print("\nClassificação das amostras de teste:")
print(results.to_string(index=False))

# -------------------------------------------------------------------------
# 7. Gráficos
# -------------------------------------------------------------------------
# Curva de erro dos dois treinamentos.
plt.figure(figsize=(10, 5))
for name, model in models.items():
    plt.plot(model.loss_curve_, label=f"{name} - MSE")
plt.xlabel("Época")
plt.ylabel("Erro quadrático médio")
plt.title("Questão 6 - Curva de erro dos treinamentos")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "q6_curva_erro.png", dpi=150)
plt.close()

# Comparação das classes previstas.
x = np.arange(len(results)) + 1
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, results["classe_T1"], width, label="T1 - pesos iniciais [0,1]")
plt.bar(x + width / 2, results["classe_T2"], width, label="T2 - pesos iniciais [0,2]")
plt.xlabel("Amostra")
plt.ylabel("Classe prevista")
plt.yticks([-1, 1], ["-1 / C1", "+1 / C2"])
plt.xticks(x)
plt.title("Questão 6 - Classes previstas pelos dois treinamentos")
plt.grid(axis="y", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "q6_comparacao_classes.png", dpi=150)
plt.close()

# -------------------------------------------------------------------------
# 8. Texto auxiliar para o relatório
# -------------------------------------------------------------------------
with open(OUT_DIR / "q6_explicacao.txt", "w", encoding="utf-8") as f:
    f.write("Questão 6 - Explicação dos treinamentos\n")
    f.write("\n")
    f.write("Foram executados dois treinamentos de uma rede PMC com 3 entradas, ")
    f.write("5 neurônios na camada oculta e 1 neurônio de saída. As funções de ativação ")
    f.write("da camada oculta e da saída foram tangente hiperbólica, pois a classe desejada ")
    f.write("assume valores -1 e +1.\n\n")
    f.write("No treinamento T1, os pesos e bias iniciais foram sorteados no intervalo [0, 1].\n")
    f.write("No treinamento T2, os pesos e bias iniciais foram sorteados no intervalo [0, 2].\n\n")
    f.write("O número de épocas pode variar porque a inicialização define o ponto inicial ")
    f.write("do algoritmo na superfície de erro. Como cada treinamento parte de uma região ")
    f.write("diferente dessa superfície, o caminho percorrido até atingir o erro mínimo também ")
    f.write("pode ser diferente.\n\n")
    f.write("Critério de parada: o treinamento foi interrompido quando o MSE ficou menor ")
    f.write("que 1e-3 ou quando o número máximo de épocas foi atingido.\n")

print("\nArquivos gerados em:", OUT_DIR)
print("Questão 6 concluída.")
