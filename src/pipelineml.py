"""
Script: pipelineml.py
Descripción:
Este script implementa un pipeline completo de Machine Learning que incluye las siguientes etapas:
1. Carga de datos desde un archivo CSV.
2. Preparación y división de los datos en entrenamiento y prueba.
3. Entrenamiento y evaluación de un modelo (RandomForestClassifier).
4. Registro del modelo y sus métricas en MLflow para su seguimiento y versionamiento.

El flujo principal se ejecuta desde la función main().
"""
# filepath: /ml-pipeline-project/ml-pipeline-project/src/pipelineml.py
import pandas as pd
from data_loader import load_data
from data_preparation import prepare_data
from model_trainer import train_model
from model_registry import register_model

def main():
    # Load data
    data = load_data("D:/Proyectos/mlflow/repositorios/introduction-to-mlflow/data/in/application_data.csv")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train model
    model, accuracy = train_model(X_train, y_train, X_test, y_test)
    n_estimators = model.n_estimators
    model_name = "RandomForestClassifier"

    # Register model with MLflow
    register_model(model, model_name, n_estimators, accuracy)

if __name__ == "__main__":
    main()
