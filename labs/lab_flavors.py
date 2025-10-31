# Tema 1: Flavors en MLflow

# Definición:
# Un "flavor" en MLflow es una forma estandarizada de empaquetar modelos para que puedan ser usados, desplegados o reutilizados en diferentes entornos. 
# MLflow soporta varios flavors, como 'python_function', 'sklearn', 'keras', 'pytorch', etc.

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Selecciona o crea el experimento
mlflow.set_experiment("lab_flavors_demo")

# Entrenamos un modelo simple
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier().fit(X, y)

# Logueamos el modelo con MLflow dentro de un run
with mlflow.start_run(run_name="flavor_example"):
    mlflow.sklearn.log_model(clf, "rf_model")

print("Modelo guardado con flavors: 'sklearn' y 'python_function'.")
print("Puedes cargarlo con mlflow.sklearn.load_model o mlflow.pyfunc.load_model.")