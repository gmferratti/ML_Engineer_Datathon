import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor


def prepare_features(raw_data):
    """Prepara os dados para treino."""
    # TODO: Implementar logica
    # 1. Fazer o split de treino e teste
    # 2. Separar features e target
    # 3. One hot encoding
    
    
    # OPCIONAL: Seleção de features removendo features com alta correlação
    # logger.info("Validating feature selection...")
    # final_feats = feature_selection(suggested_feats, df_target)
    
    return raw_data


def load_train_data():
    """Carrega os dados de treino."""
    # TODO: Implementar logica
    return pd.DataFrame(), pd.DataFrame()


# =============================
# 2) Separamos X (features) e y (TARGET)
# =============================
X = df[feature_cols].copy()
y = df["TARGET"].copy()

# =============================
# 3) Construção do Pipeline 
#    usando LightGBM (Regressor)
# =============================

# Transformer para codificar as colunas categóricas em One-Hot
one_hot_transformer = OneHotEncoder(handle_unknown="ignore")

# ColumnTransformer para aplicar one-hot apenas nas cat_cols
preprocessor = ColumnTransformer(
    transformers=[
        ("one_hot", one_hot_transformer, cat_cols)
    ],
    remainder="passthrough"  # as colunas numéricas passam sem transformação
)

# Montamos o pipeline:
#  - Preprocessamento (OneHot nas categóricas)
#  - Regressor LightGBM
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("lgbm", LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
        # Você pode ajustar mais hiperparâmetros de acordo com seu caso
    ))
])

# =============================
# 4) Split em treino e teste
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=42
)

# =============================
# 5) Treino do modelo
# =============================
model.fit(X_train, y_train)

# =============================
# 6) Avaliação do modelo
# =============================
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE (Teste): {mse:.2f}")
print(f"R^2  (Teste): {r2:.3f}")

# =============================
# 7) Exemplo de uso (predição)
# =============================
# Imagine que queremos prever o TARGET para os mesmos dados de teste
df_pred = X_test.copy()
df_pred["pred_TARGET"] = y_pred
df_pred["true_TARGET"] = y_test.values