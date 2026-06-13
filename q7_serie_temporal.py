"""
Questão 7 - Previsor Neural para Série Temporal de Potência Ativa
=================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# Configurações gerais
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Serie temporal.xlsx"
OUT_DIR = BASE_DIR / "resultados_q7"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

print("=" * 70)
print("Questão 7 - Previsor Neural para Série Temporal de Potência Ativa")
print("=" * 70)

# -------------------------------------------------------------------------
# 1. Carregamento da série temporal
# -------------------------------------------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")

raw = pd.read_excel(DATA_FILE, sheet_name="carga_eletrica", header=None)
pa_series = pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

if len(pa_series) <= 48:
    raise ValueError("A série possui poucas amostras para usar defasagem de 24 horas.")

print(f"Total de amostras da série PA: {len(pa_series)}")
print(f"PA mínima: {pa_series.min():.2f} MW | PA máxima: {pa_series.max():.2f} MW")

# -------------------------------------------------------------------------
# 2. Construção do dataset supervisionado
# -------------------------------------------------------------------------
# Para cada instante k, as entradas são PA(k-1), PA(k-2), PA(k-24), e a saída é PA(k).
X = []
y = []
k_abs = []  # índice absoluto de cada saída PA(k) na série original

for k in range(24, len(pa_series)):
    X.append([pa_series[k - 1], pa_series[k - 2], pa_series[k - 24]])
    y.append(pa_series[k])
    k_abs.append(k)

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)
k_abs = np.array(k_abs, dtype=int)

print(f"Total de pares supervisionados: {len(y)}")
print("Entradas usadas: PA(k-1), PA(k-2), PA(k-24)")
print("Saída prevista: PA(k)")

# -------------------------------------------------------------------------
# 3. Divisão temporal treino/validação/teste-cego
# -------------------------------------------------------------------------
# Em série temporal não se embaralham os dados.
n_samples = len(y)
train_end = int(n_samples * 0.70)
val_end = int(n_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
k_test = k_abs[val_end:]

test_start_abs = int(k_abs[val_end])

print("\nDivisão temporal:")
print(f"Treino     : {len(y_train)} amostras")
print(f"Validação  : {len(y_val)} amostras")
print(f"Teste-cego : {len(y_test)} amostras")
print(f"Início absoluto do teste-cego na série original: índice {test_start_abs}")

# -------------------------------------------------------------------------
# 4. Funções auxiliares
# -------------------------------------------------------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def count_parameters(input_dim: int, hidden_layers: tuple[int, ...], output_dim: int = 1) -> int:
    total = 0
    previous = input_dim
    for neurons in hidden_layers:
        total += previous * neurons + neurons
        previous = neurons
    total += previous * output_dim + output_dim
    return total


def train_candidate(arch, X_tr, y_tr, X_va, y_va):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_tr_sc = scaler_X.fit_transform(X_tr)
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    X_va_sc = scaler_X.transform(X_va)

    model = MLPRegressor(
        hidden_layer_sizes=arch,
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        tol=1e-4,
        learning_rate_init=0.001,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr_sc, y_tr_sc)

    y_va_pred_sc = model.predict(X_va_sc)
    y_va_pred = scaler_y.inverse_transform(y_va_pred_sc.reshape(-1, 1)).ravel()

    return model, scaler_X, scaler_y, y_va_pred

# -------------------------------------------------------------------------
# 5. Teste de arquiteturas candidatas
# -------------------------------------------------------------------------
candidates = [
    (5,),
    (10,),
    (15,),
    (20,),
    (10, 5),
    (15, 5),
    (20, 10),
]

print("\nTestando previsores candidatos:")
comparison_rows = []

for arch in candidates:
    model, _, _, y_val_pred = train_candidate(arch, X_train, y_train, X_val, y_val)
    row = {
        "arquitetura": str(arch),
        "parametros": count_parameters(X.shape[1], arch),
        "iteracoes": model.n_iter_,
        "RMSE_validacao": rmse(y_val, y_val_pred),
        "MAPE_validacao_percent": mape(y_val, y_val_pred),
    }
    comparison_rows.append(row)
    print(
        f"Arquitetura {str(arch):<10} | params={row['parametros']:>4} | "
        f"iter={row['iteracoes']:>4} | RMSE_val={row['RMSE_validacao']:.3f} | "
        f"MAPE_val={row['MAPE_validacao_percent']:.3f}%"
    )

comparison_df = pd.DataFrame(comparison_rows).sort_values(["RMSE_validacao", "parametros"])
comparison_df.to_csv(OUT_DIR / "q7_comparacao_arquiteturas.csv", index=False)

selected_arch_str = comparison_df.iloc[0]["arquitetura"]
selected_arch = tuple(int(s) for s in selected_arch_str.replace("(", "").replace(")", "").replace(",", " ").split())

print("\nArquitetura selecionada pelo menor RMSE de validação:", selected_arch)

# -------------------------------------------------------------------------
# 6. Treinamento final com treino + validação
# -------------------------------------------------------------------------
X_trainval = X[:val_end]
y_trainval = y[:val_end]

scaler_X_final = MinMaxScaler()
scaler_y_final = MinMaxScaler()

X_trainval_sc = scaler_X_final.fit_transform(X_trainval)
y_trainval_sc = scaler_y_final.fit_transform(y_trainval.reshape(-1, 1)).ravel()
X_test_sc = scaler_X_final.transform(X_test)

final_model = MLPRegressor(
    hidden_layer_sizes=selected_arch,
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    tol=1e-4,
    learning_rate_init=0.001,
    random_state=RANDOM_STATE,
)

print("\nTreinando modelo final com treino + validação...")
final_model.fit(X_trainval_sc, y_trainval_sc)
print(f"Treinamento final concluído em {final_model.n_iter_} épocas.")

# -------------------------------------------------------------------------
# 7. Teste-cego de 1 passo à frente
# -------------------------------------------------------------------------
y_test_pred_sc = final_model.predict(X_test_sc)
y_test_pred = scaler_y_final.inverse_transform(y_test_pred_sc.reshape(-1, 1)).ravel()

rmse_1 = rmse(y_test, y_test_pred)
mape_1 = mape(y_test, y_test_pred)

print("\nTeste-cego: previsão 1 passo à frente")
print(f"RMSE = {rmse_1:.3f} MW")
print(f"MAPE = {mape_1:.3f}%")

pd.DataFrame([{
    "arquitetura_escolhida": str(selected_arch),
    "parametros": count_parameters(X.shape[1], selected_arch),
    "iteracoes_modelo_final": final_model.n_iter_,
    "RMSE_1_passo_teste": rmse_1,
    "MAPE_1_passo_teste_percent": mape_1,
}]).to_csv(OUT_DIR / "q7_metricas_1_passo_teste.csv", index=False)

one_step_df = pd.DataFrame({
    "indice_absoluto_k": k_test,
    "PA_real": y_test,
    "PA_prevista_1_passo": y_test_pred,
    "erro": y_test - y_test_pred,
    "erro_abs": np.abs(y_test - y_test_pred),
    "erro_percent_abs": np.abs((y_test - y_test_pred) / y_test) * 100,
})
one_step_df.to_csv(OUT_DIR / "q7_previsao_1_passo_teste.csv", index=False)

# Gráfico de 1 passo à frente.
plt.figure(figsize=(12, 5))
plt.plot(one_step_df["PA_real"].to_numpy(), label="PA real")
plt.plot(one_step_df["PA_prevista_1_passo"].to_numpy(), label="PA prevista - 1 passo", alpha=0.85)
plt.xlabel("Amostra no conjunto de teste-cego")
plt.ylabel("Potência Ativa (MW)")
plt.title(f"Questão 7 - Teste-cego 1 passo à frente | MAPE={mape_1:.2f}% | RMSE={rmse_1:.2f} MW")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "q7_serie_1_passo.png", dpi=150)
plt.close()

# -------------------------------------------------------------------------
# 8. Previsão recursiva de 1 a 24 passos à frente
# -------------------------------------------------------------------------
# Nesta etapa, para prever vários passos à frente, o modelo passa a usar suas
# próprias previsões como entradas futuras. Isso permite avaliar o acúmulo de erro.
horizons = 24
origins = list(range(test_start_abs, len(pa_series) - horizons + 1, horizons))

print("\nPrevisão recursiva de 1 a 24 passos à frente:")
print(f"Total de origens de previsão no teste-cego: {len(origins)}")

metrics_by_horizon = {h: {"real": [], "pred": []} for h in range(1, horizons + 1)}
recursive_rows = []

for origin in origins:
    # Histórico real disponível até o instante imediatamente anterior à origem.
    # A partir da origem, o histórico passa a receber previsões.
    history = list(pa_series[:origin])

    for h in range(1, horizons + 1):
        k = origin + h - 1

        # Defasagens. Para h > 1, PA(k-1) e/ou PA(k-2) podem ser previsões.
        x_in = np.array([[history[-1], history[-2], history[-24]]], dtype=float)
        x_in_sc = scaler_X_final.transform(x_in)

        y_pred_sc = final_model.predict(x_in_sc)
        y_pred = scaler_y_final.inverse_transform(y_pred_sc.reshape(-1, 1))[0, 0]

        # Atualiza histórico com a previsão, caracterizando previsão recursiva.
        history.append(float(y_pred))

        y_real = float(pa_series[k])
        metrics_by_horizon[h]["real"].append(y_real)
        metrics_by_horizon[h]["pred"].append(float(y_pred))

        recursive_rows.append({
            "origem_previsao": origin,
            "horizonte_horas": h,
            "indice_absoluto_k": k,
            "PA_real": y_real,
            "PA_prevista_recursiva": float(y_pred),
            "erro": y_real - float(y_pred),
            "erro_abs": abs(y_real - float(y_pred)),
            "erro_percent_abs": abs((y_real - float(y_pred)) / y_real) * 100,
        })

# Métricas por horizonte.
metrics_rows = []
for h in range(1, horizons + 1):
    y_real_h = np.array(metrics_by_horizon[h]["real"], dtype=float)
    y_pred_h = np.array(metrics_by_horizon[h]["pred"], dtype=float)
    metrics_rows.append({
        "Horizonte_horas": h,
        "RMSE": rmse(y_real_h, y_pred_h),
        "MAPE_percent": mape(y_real_h, y_pred_h),
        "Numero_de_previsoes": len(y_real_h),
    })

metrics_h_df = pd.DataFrame(metrics_rows)
metrics_h_df.to_csv(OUT_DIR / "q7_metricas_por_horizonte.csv", index=False)

recursive_df = pd.DataFrame(recursive_rows)
recursive_df.to_csv(OUT_DIR / "q7_previsao_recursiva_24h.csv", index=False)

print(metrics_h_df.to_string(index=False))

# Gráfico das métricas por horizonte.
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(metrics_h_df["Horizonte_horas"], metrics_h_df["RMSE"], marker="o", label="RMSE")
ax2.plot(metrics_h_df["Horizonte_horas"], metrics_h_df["MAPE_percent"], marker="x", linestyle="--", label="MAPE")

ax1.set_xlabel("Horizonte de previsão (horas)")
ax1.set_ylabel("RMSE (MW)")
ax2.set_ylabel("MAPE (%)")
plt.title("Questão 7 - Erro da previsão recursiva por horizonte")
ax1.grid(True, alpha=0.35)
fig.tight_layout()
plt.savefig(OUT_DIR / "q7_mape_rmse_horizonte.png", dpi=150)
plt.close()

# Gráfico concatenado das previsões recursivas.
plt.figure(figsize=(12, 5))
plt.plot(recursive_df["PA_real"].to_numpy(), label="PA real")
plt.plot(recursive_df["PA_prevista_recursiva"].to_numpy(), label="PA prevista recursiva", alpha=0.85)
plt.xlabel("Previsões recursivas concatenadas no teste-cego")
plt.ylabel("Potência Ativa (MW)")
plt.title("Questão 7 - Previsão recursiva de 1 a 24 horas à frente")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "q7_previsao_recursiva_24h.png", dpi=150)
plt.close()

print("\nArquivos gerados em:", OUT_DIR)
print("Questão 7 concluída.")
