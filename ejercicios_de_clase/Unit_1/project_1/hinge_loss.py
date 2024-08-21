import numpy as np

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Encuentra la pérdida de margen para un único punto de datos dado unos parámetros de clasificación específicos.

    Args:
        `feature_vector` - numpy array que describe el punto de datos dado.
        `label` - float, la clasificación correcta del punto de datos.
        `theta` - numpy array que describe el clasificador lineal.
        `theta_0` - float que representa el parámetro de desplazamiento.

    Returns:
        La pérdida de margen, como un float, asociada con el punto de datos dado y los parámetros.
    """
    # Calcula la predicción del clasificador para el punto de datos dado
    prediction = np.dot(theta, feature_vector) + theta_0
    
    # Calcula la pérdida de margen
    loss = max(0, 1 - label * prediction)
    
    return loss


import numpy as np

#############################################################################################

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Encuentra la pérdida de margen para los parámetros de clasificación dados,
    promediada sobre un conjunto de datos dado.

    Args:
        `feature_matrix` - matriz de numpy que describe los datos dados. Cada fila representa un punto de datos individual.
        `labels` - arreglo de numpy donde el k-ésimo elemento del arreglo es la clasificación correcta de la k-ésima fila de la matriz de características.
        `theta` - arreglo de numpy que describe el clasificador lineal.
        `theta_0` - número real que representa el parámetro de desplazamiento.

    Returns:
        La pérdida de margen, como un float, asociada con el conjunto de datos dado y los parámetros.
        Este número debería ser la pérdida de margen promedio en todos los datos.
    """
    # Número de puntos de datos
    num_samples = feature_matrix.shape[0]
    
    # Calcular la predicción del clasificador para todos los puntos de datos
    predictions = np.dot(feature_matrix, theta) + theta_0
    
    # Calcular la pérdida de margen para cada punto de datos
    losses = np.maximum(0, 1 - labels * predictions)
    
    # Promediar la pérdida de margen
    average_loss = np.mean(losses)
    
    return average_loss
