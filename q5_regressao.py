"""
Questão 5 - Sensor Virtual de Posição com PMC (Regressão)
==========================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# Configurações gerais
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Regreção.xlsx"
OUT_DIR = BASE_DIR / "resultados_q5"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

print("=" * 70)
print("Questão 5 - Sensor Virtual Regressor para Posição do Objeto")
print("=" * 70)

# -------------------------------------------------------------------------
# 1. Carregamento dos dados
# -------------------------------------------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")

# O arquivo tem v1 na linha 0, v2 na linha 1 e y na linha 2.
df = pd.read_excel(DATA_FILE, sheet_name="ball_pt", header=None)

v1 = pd.to_numeric(df.iloc[0, 1:], errors="coerce").dropna().to_numpy(dtype=float)
v2 = pd.to_numeric(df.iloc[1, 1:], errors="coerce").dropna().to_numpy(dtype=float)
y = pd.to_numeric(df.iloc[2, 1:], errors="coerce").dropna().to_numpy(dtype=float)

n = min(len(v1), len(v2), len(y))
v1, v2, y = v1[:n], v2[:n], y[:n]

X = np.column_stack([v1, v2])
y = y.ravel()

print(f"Total de amostras: {len(y)}")
print(f"v1: min={v1.min():.4f}, max={v1.max():.4f}")
print(f"v2: min={v2.min():.4f}, max={v2.max():.4f}")
print(f"y : min={y.min():.4f}, max={y.max():.4f} polegadas")

# -------------------------------------------------------------------------
# 2. Divisão treino/validação/teste
# -------------------------------------------------------------------------
# Como os dados estão ordenados pela posição y, a separação sem embaralhar
# prejudica o teste. Aqui fazemos uma separação aleatória reprodutível.
# Proporção final: 60% treino, 20% validação, 20% teste.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, shuffle=True, random_state=RANDOM_STATE
)

print("\nDivisão dos dados:")
print(f"Treino    : {len(y_train)} amostras")
print(f"Validação : {len(y_val)} amostras")
print(f"Teste     : {len(y_test)} amostras")

# -------------------------------------------------------------------------
# 3. Normalização
# -------------------------------------------------------------------------
# Os scalers são ajustados apenas no treino para evitar vazamento de informação.
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_val_sc = scaler_X.transform(X_val)
X_test_sc = scaler_X.transform(X_test)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_sc = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

# -------------------------------------------------------------------------
# 4. Busca de arquiteturas com parcimônia
# -------------------------------------------------------------------------
def count_parameters(input_dim: int, hidden_layers: tuple[int, ...], output_dim: int = 1) -> int:
    """Conta pesos e bias de uma rede totalmente conectada."""
    total = 0
    previous = input_dim
    for neurons in hidden_layers:
        total += previous * neurons + neurons  # pesos + bias da camada
        previous = neurons
    total += previous * output_dim + output_dim
    return total

candidates = [
    (2,),
    (3,),
    (4,),
    (5,),
    (7,),
    (10,),
    (3, 2),
    (5, 3),
    (7, 3),
    (10, 5),
]

print("\nBusca de arquiteturas candidatas:")
results = []
models = {}

for arch in candidates:
    model = MLPRegressor(
        hidden_layer_sizes=arch,
        activation="tanh",
        solver="lbfgs",      # bom para bases pequenas
        max_iter=5000,
        tol=1e-6,
        random_state=1,
    )

    model.fit(X_train_sc, y_train_sc)

    y_val_pred_sc = model.predict(X_val_sc)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_sc.reshape(-1, 1)).ravel()

    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    params = count_parameters(X.shape[1], arch)

    results.append({
        "arquitetura": str(arch),
        "parametros": params,
        "iteracoes": model.n_iter_,
        "RMSE_validacao": rmse_val,
        "MAE_validacao": mae_val,
        "R2_validacao": r2_val,
    })
    models[arch] = model

    print(
        f"Arquitetura {str(arch):<10} | params={params:>3} | "
        f"iter={model.n_iter_:>4} | RMSE_val={rmse_val:.6f} | R²_val={r2_val:.6f}"
    )

results_df = pd.DataFrame(results).sort_values(["RMSE_validacao", "parametros"])
results_df.to_csv(OUT_DIR / "q5_busca_arquiteturas.csv", index=False)

# Escolha parcimoniosa:
# Primeiro encontra o melhor RMSE de validação. Depois aceita modelos até 10% piores.
# Entre esses, escolhe o de menor número de parâmetros.
best_rmse = results_df["RMSE_validacao"].min()
tolerance = 1.10 * best_rmse
eligible = results_df[results_df["RMSE_validacao"] <= tolerance].copy()
eligible = eligible.sort_values(["parametros", "RMSE_validacao"])
selected_arch_str = eligible.iloc[0]["arquitetura"]

# Converte string da tupla de volta para tupla de inteiros com segurança.
selected_arch = tuple(int(s) for s in selected_arch_str.replace("(", "").replace(")", "").replace(",", " ").split())

print("\nCritério de escolha:")
print(f"Melhor RMSE de validação = {best_rmse:.6f}")
print(f"Limite aceito pela parcimônia = {tolerance:.6f}")
print(f"Arquitetura selecionada = {selected_arch}")

# -------------------------------------------------------------------------
# 5. Treinamento final com treino + validação e avaliação no teste
# -------------------------------------------------------------------------
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

scaler_X_final = StandardScaler()
scaler_y_final = StandardScaler()

X_trainval_sc = scaler_X_final.fit_transform(X_trainval)
X_test_sc_final = scaler_X_final.transform(X_test)
y_trainval_sc = scaler_y_final.fit_transform(y_trainval.reshape(-1, 1)).ravel()

final_model = MLPRegressor(
    hidden_layer_sizes=selected_arch,
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    tol=1e-6,
    random_state=1,
)
final_model.fit(X_trainval_sc, y_trainval_sc)

y_test_pred_sc = final_model.predict(X_test_sc_final)
y_test_pred = scaler_y_final.inverse_transform(y_test_pred_sc.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\nMétricas finais no conjunto de teste:")
print(f"MSE  = {mse:.6f}")
print(f"RMSE = {rmse:.6f} polegadas")
print(f"MAE  = {mae:.6f} polegadas")
print(f"R²   = {r2:.6f}")
print(f"Iterações do modelo final = {final_model.n_iter_}")

metrics_df = pd.DataFrame([{
    "arquitetura_escolhida": str(selected_arch),
    "parametros": count_parameters(X.shape[1], selected_arch),
    "iteracoes": final_model.n_iter_,
    "MSE_teste": mse,
    "RMSE_teste": rmse,
    "MAE_teste": mae,
    "R2_teste": r2,
}])
metrics_df.to_csv(OUT_DIR / "q5_metricas_finais.csv", index=False)

pred_test_df = pd.DataFrame({
    "v1": X_test[:, 0],
    "v2": X_test[:, 1],
    "y_real": y_test,
    "y_previsto": y_test_pred,
    "erro": y_test - y_test_pred,
    "erro_abs": np.abs(y_test - y_test_pred),
}).sort_values("y_real")
pred_test_df.to_csv(OUT_DIR / "q5_previsoes_teste.csv", index=False)

# Previsão em todos os dados apenas para visualização da curva completa.
X_all_sc = scaler_X_final.transform(X)
y_all_pred_sc = final_model.predict(X_all_sc)
y_all_pred = scaler_y_final.inverse_transform(y_all_pred_sc.reshape(-1, 1)).ravel()

pred_all_df = pd.DataFrame({
    "v1": X[:, 0],
    "v2": X[:, 1],
    "y_real": y,
    "y_previsto": y_all_pred,
    "erro": y - y_all_pred,
    "erro_abs": np.abs(y - y_all_pred),
}).sort_values("y_real")
pred_all_df.to_csv(OUT_DIR / "q5_previsoes_todos_os_dados.csv", index=False)

# -------------------------------------------------------------------------
# 6. Gráficos
# -------------------------------------------------------------------------
# Curva y real e previsto, ordenada por posição real.
plt.figure(figsize=(11, 5))
plt.plot(pred_all_df["y_real"].to_numpy(), label="Posição real y", marker="o", linewidth=1)
plt.plot(pred_all_df["y_previsto"].to_numpy(), label="Posição prevista pela PMC", marker="x", linestyle="--", linewidth=1)
plt.xlabel("Amostras ordenadas pela posição real")
plt.ylabel("Posição y (polegadas)")
plt.title("Questão 5 - Sensor virtual: posição real x posição prevista")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "q5_real_vs_previsto_todos.png", dpi=150)
plt.close()

# Scatter no conjunto de teste.
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, edgecolors="k", alpha=0.8)
lo = min(y_test.min(), y_test_pred.min())
hi = max(y_test.max(), y_test_pred.max())
plt.plot([lo, hi], [lo, hi], "--", label="Previsão ideal")
plt.xlabel("y real (polegadas)")
plt.ylabel("y previsto (polegadas)")
plt.title(f"Questão 5 - Teste: real x previsto (R²={r2:.4f})")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "q5_scatter_teste.png", dpi=150)
plt.close()

print("\nArquivos gerados em:", OUT_DIR)
print("Questão 5 concluída.")
