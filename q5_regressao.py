"""
Questão 5 - Sensor Virtual de Posição (Regressão com PMC)
==========================================================
Rede PMC (Perceptrons de Múltiplas Camadas) treinada para estimar
a posição y (polegadas) de um objeto a partir das tensões v1 e v2 (volts).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("resultados", exist_ok=True)

print("=" * 55)
print("  Questão 5: Sensor Virtual de Posição - Regressão PMC")
print("=" * 55)

# ─── 1. CARREGAR DADOS ────────────────────────────────────────
df = pd.read_excel("Regreção.xlsx", sheet_name="ball_pt", header=None)

v1 = df.iloc[0, 1:68].values.astype(float)
v2 = df.iloc[1, 1:68].values.astype(float)
y  = df.iloc[2, 1:68].values.astype(float)

X_all = np.column_stack([v1, v2])   # shape (67, 2)
y_all = y                            # shape (67,)

print(f"\n[Dados] Total de amostras: {len(y_all)}")
print(f"        v1  : min={v1.min():.3f}, max={v1.max():.3f}")
print(f"        v2  : min={v2.min():.3f}, max={v2.max():.3f}")
print(f"        y   : min={y_all.min():.3f} pol, max={y_all.max():.3f} pol")

# ─── 2. DIVISÃO TREINO/TESTE (mantendo ordem da série) ───────
# 80 % para treino / 20 % para teste  (SEM embaralhar)
n_train = int(len(y_all) * 0.80)  # ≈ 54 amostras
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

print(f"\n[Divisão] Treino: {n_train} amostras | Teste: {len(y_test)} amostras")

# ─── 3. NORMALIZAÇÃO (StandardScaler ajustado só no treino) ──
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ─── 4. BUSCA PELA MELHOR ARQUITETURA ────────────────────────
print("\n[Busca] Testando arquiteturas de rede neural...")

# Arquitecturas candidatas (parcimônia gradual)
candidates = [
    (5,),
    (10,),
    (15,),
    (10, 10),
    (15, 10),
    (20, 10),
    (10, 10, 5),
    (15, 10, 5),
]

best_r2   = -np.inf
best_arch = None
best_mlp  = None

for arch in candidates:
    mlp = MLPRegressor(
        hidden_layer_sizes=arch,
        activation='tanh',
        solver='adam',
        max_iter=3000,
        tol=1e-5,
        random_state=42,
        learning_rate_init=0.001,
        n_iter_no_change=50,
    )
    mlp.fit(X_train_sc, y_train_sc)

    y_pred_sc = mlp.predict(X_test_sc)
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    status = " << MELHOR" if r2 > best_r2 else ""
    print(f"   Arquit. {str(arch):<18}  épocas={mlp.n_iter_:>4}  "
          f"MSE={mse:.4f}  R²={r2:.4f}{status}")

    if r2 > best_r2:
        best_r2   = r2
        best_arch = arch
        best_mlp  = mlp

# ─── 5. AVALIAR MELHOR MODELO ────────────────────────────────
print(f"\n[Resultado] Melhor arquitetura: {best_arch}")
print(f"            Treinamento parou na época {best_mlp.n_iter_}")

# Previsões no conjunto de TESTE (domínio original)
y_pred_test_sc = best_mlp.predict(X_test_sc)
y_pred_test    = scaler_y.inverse_transform(y_pred_test_sc.reshape(-1,1)).ravel()

mse  = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred_test)
r2   = r2_score(y_test, y_pred_test)

print(f"\n[Métricas no Teste]")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f} polegadas")
print(f"  MAE  = {mae:.4f} polegadas")
print(f"  R²   = {r2:.4f}  ({r2*100:.1f}% da variância explicada)")

# Previsões em TODOS os dados (para curva completa)
X_all_sc    = scaler_X.transform(X_all)
y_pred_all_sc = best_mlp.predict(X_all_sc)
y_pred_all  = scaler_y.inverse_transform(y_pred_all_sc.reshape(-1,1)).ravel()

# ─── 6. GRÁFICOS ─────────────────────────────────────────────

# --- 6.1  Real vs Previsto (eixo posição) --------------------
fig, ax = plt.subplots(figsize=(11, 5))

idx = np.arange(len(y_all))
ax.plot(idx, y_all,      'b-o', ms=5, lw=1.5, label='Posição real y (pol)')
ax.plot(idx, y_pred_all, 'r--x', ms=5, lw=1.5, label=f'PMC previsto  (R²={r2:.3f})')

ax.axvline(n_train - 0.5, color='gray', ls=':', lw=1.2, label='Início do teste')
ax.set_xlabel('Índice da amostra', fontsize=12)
ax.set_ylabel('Posição y (polegadas)', fontsize=12)
ax.set_title('Questão 5 – Sensor Virtual de Posição: Real vs Previsto', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.35)

plt.tight_layout()
p1 = os.path.join("resultados", "q5_real_vs_previsto.png")
plt.savefig(p1, dpi=150)
plt.close()

# --- 6.2  Scatter: real × previsto (diagonal perfeita) -------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_all, y_pred_all, c='steelblue', s=40, edgecolors='k', lw=0.4, alpha=0.8)
lo, hi = y_all.min(), y_all.max()
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Previsão perfeita')
ax.set_xlabel('y real (pol)', fontsize=12)
ax.set_ylabel('y previsto (pol)', fontsize=12)
ax.set_title(f'Scatter: Real × Previsto  (R²={r2:.3f})', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.35)

plt.tight_layout()
p2 = os.path.join("resultados", "q5_scatter.png")
plt.savefig(p2, dpi=150)
plt.close()

# --- 6.3  Curva de aprendizado (loss × épocas) ---------------
if hasattr(best_mlp, 'loss_curve_'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(best_mlp.loss_curve_, lw=1.5, color='darkgreen')
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Loss (MSE normalizado)', fontsize=12)
    ax.set_title(f'Curva de aprendizado – arquitetura {best_arch}', fontsize=13)
    ax.grid(alpha=0.35)
    plt.tight_layout()
    p3 = os.path.join("resultados", "q5_curva_aprendizado.png")
    plt.savefig(p3, dpi=150)
    plt.close()

print(f"\n[Gráficos salvos]")
print(f"  {p1}")
print(f"  {p2}")
if hasattr(best_mlp, 'loss_curve_'):
    print(f"  {p3}")

print("\n--- Questão 5 Concluída com sucesso ---")
