def register_model(model, model_name, n_estimators, accuracy):
    """
    Registra un modelo entrenado en MLflow junto con sus parámetros y métricas.

    Este procedimiento inicia una ejecución (`run`) en un servidor MLflow local
    y guarda el modelo entrenado como artefacto, además de registrar los parámetros
    y la métrica de precisión obtenida. Esto permite rastrear experimentos y
    comparar resultados desde la interfaz de MLflow.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Modelo de machine learning previamente entrenado que se desea registrar.
    model_name : str
        Nombre del modelo a registrar (por ejemplo, "RandomForestClassifier").
    n_estimators : int
        Número de estimadores (árboles) utilizados en el modelo.
    accuracy : float
        Precisión del modelo sobre el conjunto de prueba (valor entre 0 y 1).

    Returns
    -------
    None
        La función no retorna ningún valor; su propósito es registrar la información
        en el servidor de MLflow configurado.

    Note
    -----
    - Los artefactos del modelo se almacenarán en la ruta configurada por MLflow.
    """
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Mi primer Modelo")

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
