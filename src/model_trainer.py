from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo de clasificación utilizando Random Forest y evalúa su precisión.

    Este procedimiento crea un modelo `RandomForestClassifier` con 50 árboles, 
    lo entrena con los datos de entrenamiento y calcula la precisión (accuracy) 
    sobre el conjunto de prueba.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Conjunto de características (features) para el entrenamiento del modelo.
    y_train : pandas.Series or numpy.ndarray
        Variable objetivo correspondiente a `X_train`.
    X_test : pandas.DataFrame or numpy.ndarray
        Conjunto de características (features) para la evaluación del modelo.
    y_test : pandas.Series or numpy.ndarray
        Variable objetivo correspondiente a `X_test`.

    Returns
    -------
    clf : sklearn.ensemble.RandomForestClassifier
        Modelo de Random Forest entrenado.
    accuracy : float
        Precisión del modelo sobre el conjunto de prueba (valores entre 0 y 1).
    """
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, accuracy