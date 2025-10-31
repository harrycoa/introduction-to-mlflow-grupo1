# Tema 2: Model Signatures

# Definición:
# La 'signature' de un modelo en MLflow describe el esquema de entrada y salida esperado del modelo.
# Esto ayuda a validar los datos en producción y evitar errores por incompatibilidad de formatos.

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from mlflow.models.signature import infer_signature

# Selecciona o crea el experimento
mlflow.set_experiment("lab_model_signatures_demo")

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier().fit(X, y)

# Inferimos la signature automáticamente
signature = infer_signature(X, clf.predict(X))

# Logueamos el modelo con signature dentro de un run
with mlflow.start_run(run_name="model_signatures_example"):
    mlflow.sklearn.log_model(clf, "rf_model", signature=signature)

print("Modelo guardado con signature inferida.")
print("La signature ayuda a validar los datos de entrada y salida del modelo.")