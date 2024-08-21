import numpy as np
import matplotlib.pyplot as plt

# Definimos el rango de valores para x1
x1 = np.linspace(-5, 5, 400)

# Definimos las tres líneas
line1 = x1        # x2 = x1
line2 = x1 + 3    # x2 = x1 + 3
line_solution = x1 + 1.5  # Ejemplo de línea solución válida (x2 = x1 + 1.5)

# Definimos algunos puntos de ejemplo
points = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
labels = np.array([1, 1, -1, -1, -1])  # Etiquetas correspondientes a los puntos

# Crear la gráfica
plt.figure(figsize=(8, 8))

# Graficar las líneas
plt.plot(x1, line1, 'r--', label=r'$x_2 = x_1$')
plt.plot(x1, line2, 'b--', label=r'$x_2 = x_1 + 3$')
plt.plot(x1, line_solution, 'g-', label=r'$x_2 = x_1 + 1.5$ (Solución)')

# Graficar los puntos
for i, point in enumerate(points):
    if labels[i] == 1:
        plt.plot(point[0], point[1], 'bo')  # Puntos positivos (azul)
    else:
        plt.plot(point[0], point[1], 'rx')  # Puntos negativos (rojo)

# Etiquetas de los ejes y título
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Representación de Líneas y Puntos de Clasificación')

# Limitar el rango de los ejes
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Mostrar leyenda
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()
