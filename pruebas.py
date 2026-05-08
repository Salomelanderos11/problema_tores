import numpy as np
import time
import matplotlib.pyplot as plt


# -----------------------------------------
# parametros
# -----------------------------------------

x_min = 0
x_max = 100

y_min = 0
y_max = 100

max_torres = 10
radio_max = 20

peso_cobertura = 1.0
peso_costo = 0.5
peso_solapamiento = 0.8

penalizacion_fuera = 1000
max_solapamiento = 15

resolucion = 100


# malla
linea_x = np.linspace(x_min, x_max, resolucion)
linea_y = np.linspace(y_min, y_max, resolucion)
print(linea_y)
print(linea_x)


grid_x, grid_y = np.meshgrid(linea_x, linea_y)

print("")
print("")
print(grid_x)