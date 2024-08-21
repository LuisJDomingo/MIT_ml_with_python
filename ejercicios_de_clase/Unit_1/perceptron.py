import numpy as np

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Actualiza los parámetros de clasificación `theta` y `theta_0` mediante un solo
    paso del algoritmo Perceptrón. Devuelve los nuevos parámetros en lugar de modificar
    en el lugar.

    Args:
        feature_vector - Un arreglo de numpy que describe un solo punto de datos.
        label - La clasificación correcta del vector de características.
        current_theta - El theta actual utilizado por el algoritmo Perceptrón antes de esta actualización.
        current_theta_0 - El theta_0 actual utilizado por el algoritmo Perceptrón antes de esta actualización.

    Returns:
        Una tupla que contiene dos valores:
        - El parámetro de coeficiente de características actualizado `theta` como un arreglo de numpy.
        - El parámetro de desplazamiento actualizado `theta_0` como un número flotante.
    """
    # Calcular la predicción del modelo para el punto de datos dado
    prediction = np.dot(current_theta, feature_vector) + current_theta_0
    
    # Calcular el error
    error = label * prediction
    
    # Si la predicción es incorrecta (error <= 0), actualizamos los parámetros
    if error <= 0:
        new_theta = current_theta + label * feature_vector
        new_theta_0 = current_theta_0 + label
    else:
        # No es necesario actualizar si la predicción es correcta
        new_theta = current_theta
        new_theta_0 = current_theta_0
    
    return new_theta, new_theta_0

##################################################################################################

import numpy as np

def perceptron(feature_matrix, labels, T):
    """
    Ejecuta el algoritmo Perceptrón completo en un conjunto de datos dado. Realiza T
    iteraciones a través del conjunto de datos: no se detiene antes.

    Args:
        `feature_matrix` - matriz de numpy que describe los datos dados. Cada fila representa un punto de datos individual.
        `labels` - arreglo de numpy donde el k-ésimo elemento del arreglo es la clasificación correcta de la k-ésima fila de la matriz de características.
        `T` - entero que indica cuántas veces el algoritmo Perceptrón debe iterar a través de la matriz de características.

    Returns:
        Una tupla que contiene dos valores:
        - El parámetro de coeficiente de características `theta` como un arreglo de numpy (encontrado después de T iteraciones a través de la matriz de características).
        - El parámetro de desplazamiento `theta_0` como un número flotante (encontrado también después de T iteraciones a través de la matriz de características).
    """
    # Inicializar los parámetros
    num_features = feature_matrix.shape[1]
    theta = np.zeros(num_features)
    theta_0 = 0.0
    
    # Ejecutar T iteraciones
    for _ in range(T):
        # Iterar a través de todos los puntos de datos
        for i in range(feature_matrix.shape[0]):
            feature_vector = feature_matrix[i]
            label = labels[i]
            
            # Actualizar los parámetros utilizando la función perceptron_single_step_update
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
    
    return theta, theta_0

################################################################################################

import numpy as np

def average_perceptron(feature_matrix, labels, T):
    """
    Ejecuta el algoritmo Perceptrón promedio en un conjunto de datos dado. Realiza `T`
    iteraciones a través del conjunto de datos (no se detiene antes) y, por lo tanto,
    promedia sobre `T` muchos valores de parámetros.

    Args:
        `feature_matrix` - Una matriz de numpy que describe los datos dados. Cada fila representa un punto de datos individual.
        `labels` - Un arreglo de numpy donde el k-ésimo elemento del arreglo es la clasificación correcta de la k-ésima fila de la matriz de características.
        `T` - Un entero que indica cuántas veces el algoritmo Perceptrón debe iterar a través de la matriz de características.

    Returns:
        Una tupla que contiene dos valores:
        - El parámetro de coeficiente de características promedio `theta` como un arreglo de numpy (promediado sobre T iteraciones a través de la matriz de características).
        - El parámetro de desplazamiento promedio `theta_0` como un número flotante (promediado también sobre T iteraciones a través de la matriz de características).
    """
    # Inicializar los parámetros
    num_features = feature_matrix.shape[1]
    theta = np.zeros(num_features)
    theta_0 = 0.0

    # Inicializar acumuladores para el promedio
    theta_sum = np.zeros(num_features)
    theta_0_sum = 0.0

    # Ejecutar T iteraciones
    for t in range(1, T + 1):
        # Iterar a través de todos los puntos de datos
        for i in range(feature_matrix.shape[0]):
            feature_vector = feature_matrix[i]
            label = labels[i]

            # Actualizar los parámetros utilizando la función perceptron_single_step_update
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
        
        # Acumulando los valores para el promedio
        theta_sum += theta
        theta_0_sum += theta_0

    # Calcular el promedio de los parámetros
    avg_theta = theta_sum / T
    avg_theta_0 = theta_0_sum / T

    return avg_theta, avg_theta_0
