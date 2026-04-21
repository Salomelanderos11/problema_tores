import numpy as np

# ==========================================
# PASO 1: CONSTRUIR EL "MUNDO" Y LAS REGLAS
# ==========================================

# 1. Límites geográficos de la ciudad (Nuestro plano de 100x100)
X_MIN, X_MAX = 0, 100  
Y_MIN, Y_MAX = 0, 100  

# 2. Restricciones técnicas de la empresa de telecomunicaciones
N_MAX = 10     # Presupuesto máximo: 10 torres permitidas en total
R_MAX = 20     # Límite técnico: Ninguna torre puede emitir señal a más de 20 de radio

# 3. Las "Reglas del Juego" (Pesos de la función objetivo)
# Con esto le diremos a las partículas qué es bueno y qué es malo
W_COB = 1.0     # PREMIO: 1 punto por cada metro cuadrado con señal
W_COSTO = 0.5   # CASTIGO: Te restamos 0.5 puntos por lo que cueste la antena
W_SOLAP = 0.8   # CASTIGO: Te restamos 0.8 puntos si dos antenas cubren la misma calle
W_FUERA = 1000  # MULTA: Te quitamos 1000 puntos si pones una antena fuera de la ciudad

# 4. El "Escáner" (Nuestra malla virtual para calcular áreas con NumPy puro)
# Dividimos el mapa de 100x100 en 10,000 puntitos (píxeles). 
RESOLUCION = 100
x_line = np.linspace(X_MIN, X_MAX, RESOLUCION)
y_line = np.linspace(Y_MIN, Y_MAX, RESOLUCION)
X_GRID, Y_GRID = np.meshgrid(x_line, y_line)
AREA_POR_PUNTO = ((X_MAX - X_MIN) / RESOLUCION) * ((Y_MAX - Y_MIN) / RESOLUCION)
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
    
    
