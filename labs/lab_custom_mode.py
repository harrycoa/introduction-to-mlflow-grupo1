# Tema 4: Custom Model

# Definición:
# Un 'custom model' en MLflow es un modelo definido por el usuario usando la clase PythonModel.
# Permite encapsular cualquier lógica de predicción, incluso fuera de los frameworks soportados por defecto.

import mlflow
import mlflow.pyfunc
import pandas as pd

# Selecciona o crea el experimento
mlflow.set_experiment("lab_custom_model_demo")

# Definimos un modelo personalizado que suma las columnas de entrada
class MyAdderModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input.sum(axis=1)

# Logueamos el modelo custom dentro de un run
with mlflow.start_run(run_name="custom_model_example") as run:
    mlflow.pyfunc.log_model(
        artifact_path="adder_model",
        python_model=MyAdderModel()
    )
    run_id = run.info.run_id  # Guarda el run_id aquí

print("Modelo custom guardado con flavor 'python_function'.")
print("Puedes cargarlo con mlflow.pyfunc.load_model y usarlo como cualquier modelo MLflow.")

# Validación del modelo custom: cargar y predecir
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/adder_model")

# Creamos un DataFrame de prueba
df_test = pd.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [7, 8, 9]
})

# El modelo debe devolver la suma por fila
print("Predicción del modelo custom (suma por fila):")
print(df_test)
print("Resultado:", loaded_model.predict(df_test))