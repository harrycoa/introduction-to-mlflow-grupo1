# Tema 5: Model Evaluation

# Definición:
# La evaluación de modelos en MLflow permite calcular métricas y generar artefactos automáticamente,
# usando la función mlflow.evaluate. Esto facilita la comparación y validación de modelos.

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Selecciona o crea el experimento
mlflow.set_experiment("lab_model_evaluation_demo")

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names  # Nombres de las columnas

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier().fit(X_train, y_train)

# Prepara el DataFrame de test con los nombres de las features y la columna objetivo
df_test = pd.DataFrame(X_test, columns=feature_names)
df_test["target"] = y_test

with mlflow.start_run(run_name="model_evaluation_example") as run:
    mlflow.sklearn.log_model(clf, "rf_model")
    run_id = run.info.run_id

    # Evaluamos el modelo usando mlflow.evaluate
    eval_results = mlflow.evaluate(
        model=f"runs:/{run_id}/rf_model",
        data=df_test,
        targets="target",  # Nombre de la columna objetivo
        model_type="classifier"
    )

    print("Métricas de evaluación:", eval_results.metrics)
    print("mlflow.evaluate calcula automáticamente métricas y artefactos para el modelo.")