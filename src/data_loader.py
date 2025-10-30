def load_data(file_path):
    """
    Carga un conjunto de datos desde un archivo CSV y lo devuelve como un DataFrame de pandas.

    Parameters
    ----------
    file_path : str
        Ruta completa o relativa al archivo CSV que se desea leer.

    Returns
    -------
    pandas.DataFrame
        DataFrame que contiene los datos leídos del archivo CSV.
    """
    import pandas as pd
    df = pd.read_csv(file_path)
    return df