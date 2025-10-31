# Tema 3: Input Examples

# Definición:
# Un 'input_example' en MLflow es un ejemplo real de los datos de entrada que espera el modelo.
# Esto facilita la validación, el testing y la documentación del modelo.

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Selecciona o crea el experimento
mlflow.set_experiment("lab_input_examples_demo")

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier().fit(X, y)

# Usamos las primeras dos filas como ejemplo de entrada
input_example = X[:2]

# Logueamos el modelo con input_example dentro de un run
with mlflow.start_run(run_name="input_examples_example"):
    mlflow.sklearn.log_model(clf, "rf_model", input_example=input_example)

print("Modelo guardado con input_example.")
print("Esto permite a otros usuarios ver cómo debe ser el input del modelo.")