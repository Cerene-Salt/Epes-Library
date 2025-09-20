# Examples

Este exemplo mostra o fluxo completo de uma auditoria com a biblioteca **epes**:
- Criar um *probe dataset* para uma feature
- Gerar previsões de dois modelos
- Calcular as métricas (ϝ, δϝ, ϝ\*)
- Visualizar os resultados com um *slice plot*

---

## 🔍 Exemplo: Comparando dois modelos simples

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from epes.probe import make_probe_df
from epes.metrics import epe_metrics
from epes.plots import slice_plot

# 1. Criar dataset de treino simples
X_train = pd.DataFrame({
    "x": np.linspace(0, 10, 50),
    "z": np.random.choice(["A", "B"], size=50)
})
y_train = 2 * X_train["x"] + np.random.normal(0, 1, size=50)

# 2. Treinar dois modelos diferentes
model_a = LinearRegression().fit(X_train[["x"]], y_train)
model_b = LinearRegression().fit(X_train[["x"]], y_train + 0.5*X_train["x"])

# 3. Criar probe dataset variando apenas "x"
probe_df = make_probe_df(X_train, feature="x", domain_min=0, domain_max=10, num_points=100)

# 4. Gerar previsões dos dois modelos
y_hat_a = model_a.predict(probe_df[["x"]])
y_hat_b = model_b.predict(probe_df[["x"]])

# 5. Calcular métricas Epe Piancé
phi, dphi, phi_star = epe_metrics(model_a.coef_, model_b.coef_)
print("ϝ =", phi, "δϝ =", dphi, "ϝ* =", phi_star)

# 6. Visualizar com slice plot
slice_plot(probe_df["x"], y_hat_a, y_hat_b, feature="x", labels=("Model A", "Model B"))


---

## 🔜 Próximos passos
- Esse exemplo fecha a **Fase 1 (Fortify the Core + Documentação)**.  
- A partir daqui, podemos iniciar a **Fase 2 (CI/CD)**: configurar GitHub Actions para rodar testes automaticamente e publicar a doc no GitHub Pages.  
