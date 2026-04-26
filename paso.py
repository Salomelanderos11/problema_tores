import numpy as np

#limites del mapa

x_min=0
x__max = 100

y_min=0
y__max = 100

max_t = 10
max_r = 20

w_cob = 1.0
w_costo = 0.5
w_solap = 0.8 
w_fuera = 1000

resolucion = 100

linea_x = np.linspace(x_min,x__max,resolucion)
lineay = np.linspace(y_min, y__max,resolucion)

gridx, gridY = np.meshgrid(x_line,y_line)
area = ((x__max -x_min)/ resolucion) * ((y__max-y_min)/resolucion)


def calcular_fitnes(vector_individuo):
    torres = np.reshape(vector_individuo,(max_t,3))

    costo_total = 0
    penalizacio_frontera = 0

    mapa_cobertura = np.zeros((resolucion,resolucion))

    for torre in torres:
        x,y,r = torre

        if r < 0.5:
            continue

        r = min(r,max_r)

        if not (x_min <= x <= x__max) or not (y_min <= y <= y__max):
            penalizacio_frontera += w_fuera
            continue

        costo_total += np.pi * (r**2)
        distanciacuadrada = (gridx - x)**2 + (gridY-y)**2
        mapa_cobertura += (distanciacuadrada <= (r**2))
        areacubierta = np.sum(mapa_cobertura>= 1) * area
        solapamiento = np.sum(np.maximum(0,mapa_cobertura - 1)) * area
        fitnes = (w_cob * areacubierta)- (w_costo*costo_total)-(w_solap-solapamiento) - penalizacio_frontera

        return fitnes


def optimizacion_pso(num_particulas=30,iteraciones=50): 
    dim = max_t * 3

    w= 0.7
    cognitivo = 1.5
    social = 1.5

    posiciones = np.random.uniform(low=0,high=100,size=(num_particulas,dim))

    for i in range(num_particulas):
        posiciones[i,2::3] = np.random.uniform(0,max_r,max_t)
    
    
