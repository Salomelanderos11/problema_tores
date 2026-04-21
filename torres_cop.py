import numpy as np  # Librería principal para cálculo matricial y vectorial súper rápido
import time         # Para medir cuánto tarda cada algoritmo en ejecutarse
import os           # Para interactuar con el sistema operativo (limpiar la consola)

# ==========================================
# 1. PARÁMETROS GLOBALES Y MALLA VIRTUAL
# ==========================================
# Límites del "mapa" o ciudad donde se pueden poner las torres
X_MIN, X_MAX = 0, 100  
Y_MIN, Y_MAX = 0, 100  
N_MAX = 10             # Cantidad máxima de torres que un algoritmo puede intentar colocar
R_MAX = 20             # Radio máximo de cobertura que puede tener una sola torre

# Pesos de la función objetivo (Determinan qué castigamos y qué premiamos)
W_COB = 1.0     # Premio multiplicador por cada metro cuadrado de cobertura
W_COSTO = 0.5   # Castigo multiplicador por el costo de construir torres grandes
W_SOLAP = 0.8   # Castigo por empalmar señales (ineficiencia)
W_FUERA = 1000  # Multa gigantesca si un algoritmo intenta poner una torre fuera de la ciudad

# --- Malla virtual para el cálculo vectorial con NumPy ---
# Para no usar librerías geométricas externas, dividimos el mapa en 10,000 "píxeles" (100x100).
RESOLUCION = 100
# Creamos ejes X y Y con 100 puntos equidistantes
x_line = np.linspace(X_MIN, X_MAX, RESOLUCION)
y_line = np.linspace(Y_MIN, Y_MAX, RESOLUCION)
# meshgrid crea una cuadrícula 2D con las coordenadas de todos esos puntos
X_GRID, Y_GRID = np.meshgrid(x_line, y_line)
# Calculamos cuánto mide (en área) cada uno de esos puntos o píxeles
AREA_POR_PUNTO = ((X_MAX - X_MIN) / RESOLUCION) * ((Y_MAX - Y_MIN) / RESOLUCION)

# ==========================================
# 2. FUNCIÓN OBJETIVO COMPARTIDA (FITNESS)
# ==========================================
# Esta función es el "juez". Todos los algoritmos le mandan sus vectores y ella les da una calificación.
def calcular_fitness(vector_individuo):
    # El vector entra plano: [x1, y1, r1, x2, y2, r2...]. Lo convertimos en una matriz de N filas y 3 columnas.
    torres = np.reshape(vector_individuo, (N_MAX, 3))
    costo_total = 0
    penalizacion_frontera = 0
    
    # Matriz 2D de ceros del tamaño del mapa. Aquí sumaremos cuántas señales llegan a cada píxel.
    mapa_cobertura = np.zeros((RESOLUCION, RESOLUCION))
    
    for torre in torres:
        x, y, r = torre  # Desempaquetamos los datos de la torre actual
        
        # Si el algoritmo le puso un radio muy chiquito, asumimos que decidió no construirla.
        if r < 0.5: continue 
        
        # Obligamos a que el radio no pase del máximo permitido técnico
        r = min(r, R_MAX)
        
        # Si las coordenadas se salen del mapa, aplicamos la súper multa y pasamos a la siguiente torre
        if not (X_MIN <= x <= X_MAX) or not (Y_MIN <= y <= Y_MAX):
            penalizacion_frontera += W_FUERA
            continue

        # El costo de la torre es proporcional al área del círculo (Pi * r^2)
        costo_total += np.pi * (r**2)
        
        # Teorema de Pitágoras: Calculamos la distancia desde la torre (x,y) a TODOS los puntos del mapa a la vez
        distancia_cuadrada = (X_GRID - x)**2 + (Y_GRID - y)**2
        
        # Donde la distancia sea menor o igual al radio de la torre, sumamos un 1 (llegó la señal)
        mapa_cobertura += (distancia_cuadrada <= (r**2))

    # Cobertura real: Contamos cuántos píxeles tienen valor >= 1 (tienen señal) y lo multiplicamos por su área
    area_cubierta = np.sum(mapa_cobertura >= 1) * AREA_POR_PUNTO
    
    # Solapamiento: A los píxeles que tienen valor > 1, les restamos 1. Ejemplo: Si llegan 3 señales, hay 2 solapamientos.
    solapamiento = np.sum(np.maximum(0, mapa_cobertura - 1)) * AREA_POR_PUNTO

    # La fórmula final matemática. Queremos MAXIMIZAR este valor.
    return (W_COB * area_cubierta) - (W_COSTO * costo_total) - (W_SOLAP * solapamiento) - penalizacion_frontera

# ==========================================
# 3. ALGORITMOS BIOINSPIRADOS
# ==========================================

# --- PARTICLE SWARM OPTIMIZATION (PSO) ---
def optimizacion_pso(num_particulas, iteraciones):
    dim = N_MAX * 3  # Dimensión del problema (3 variables por torre)
    
    # Parámetros estándar de PSO
    w, c1, c2 = 0.7, 1.5, 1.5  # Inercia (w), factor cognitivo (c1) y factor social (c2)
    
    # 1. Inicializar posiciones aleatorias de las partículas en el mapa (0 a 100)
    posiciones = np.random.uniform(low=0, high=100, size=(num_particulas, dim))
    # Ajustamos específicamente la columna de radios para que no empiecen en 100, sino entre 0 y R_MAX
    for i in range(num_particulas): posiciones[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            
    # Inicializar velocidades en cero
    velocidades = np.zeros((num_particulas, dim))
    
    # Memoria personal (PBest) y global (GBest)
    pbest_posiciones = np.copy(posiciones)
    pbest_fitness = np.array([calcular_fitness(p) for p in posiciones])
    
    gbest_idx = np.argmax(pbest_fitness) # Índice de la mejor partícula global
    gbest_posicion = np.copy(pbest_posiciones[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    # Ciclo de evolución
    for it in range(iteraciones):
        # Vectores aleatorios para darle estocasticidad al movimiento
        r1, r2 = np.random.rand(num_particulas, dim), np.random.rand(num_particulas, dim)
        
        # Ecuación de movimiento de PSO: Inercia + Memoria propia + Memoria de la bandada
        velocidades = w * velocidades + c1 * r1 * (pbest_posiciones - posiciones) + c2 * r2 * (gbest_posicion - posiciones)
        posiciones += velocidades  # Mover partículas
        
        for i in range(num_particulas):
            for j in range(N_MAX):
                # np.clip obliga a que los valores no se salgan de los límites [min, max]
                posiciones[i, j*3]     = np.clip(posiciones[i, j*3], X_MIN, X_MAX)     # X
                posiciones[i, j*3+1]   = np.clip(posiciones[i, j*3+1], Y_MIN, Y_MAX)   # Y
                posiciones[i, j*3+2]   = np.clip(posiciones[i, j*3+2], 0, R_MAX)       # Radio
            
            fit = calcular_fitness(posiciones[i])
            
            # Actualizar memoria personal si mejoró
            if fit > pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_posiciones[i] = np.copy(posiciones[i])
                # Actualizar memoria global si superó al líder
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_posicion = np.copy(posiciones[i])
                    
    return gbest_posicion, gbest_fitness

# --- ALGORITMO GENÉTICO (GA) ---
def optimizacion_genetica(tam_poblacion, generaciones):
    dim = N_MAX * 3  
    
    # 1. Crear población inicial aleatoria (Padres)
    poblacion = np.random.uniform(low=0, high=100, size=(tam_poblacion, dim))
    for i in range(tam_poblacion): poblacion[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            
    mejor_individuo, mejor_fitness = None, -float('inf')

    # Ciclo de generaciones
    for gen in range(generaciones):
        # Evaluar a toda la población
        fitness_pob = np.array([calcular_fitness(ind) for ind in poblacion])
        
        # Buscar al mejor de la generación
        idx_mejor = np.argmax(fitness_pob)
        if fitness_pob[idx_mejor] > mejor_fitness:
            mejor_fitness = fitness_pob[idx_mejor]
            mejor_individuo = np.copy(poblacion[idx_mejor])
            
        nueva_poblacion = np.zeros_like(poblacion)
        # ELITISMO: El mejor individuo pasa directo a la siguiente generación sin ser alterado
        nueva_poblacion[0] = np.copy(mejor_individuo) 
        
        # Crear los hijos restantes
        for i in range(1, tam_poblacion):
            # SELECCIÓN POR TORNEO: Escogemos 3 al azar y nos quedamos con el mejor. Hacemos esto 2 veces para tener 2 padres.
            p1 = poblacion[np.random.choice(tam_poblacion, 3)[np.argmax(fitness_pob[np.random.choice(tam_poblacion, 3)])]]
            p2 = poblacion[np.random.choice(tam_poblacion, 3)[np.argmax(fitness_pob[np.random.choice(tam_poblacion, 3)])]]
            
            # CRUCE UNIFORME: 50% de probabilidad de heredar el gen del padre 1, 50% del padre 2
            hijo = np.where(np.random.rand(dim) > 0.5, p1, p2)
            
            # MUTACIÓN: 15% de probabilidad de que el hijo sufra una mutación en sus genes
            if np.random.rand() < 0.15: 
                # Sumamos ruido aleatorio basado en distribución normal (Gaussiana)
                hijo[0::3] += np.random.normal(0, 5, N_MAX) # Mutar coordenadas X
                hijo[1::3] += np.random.normal(0, 5, N_MAX) # Mutar coordenadas Y
                hijo[2::3] += np.random.normal(0, 2, N_MAX) # Mutar Radios
            
            # Reparar fenotipo: Asegurar que la mutación no sacó a la torre de la ciudad
            for j in range(N_MAX):
                hijo[j*3]   = np.clip(hijo[j*3], X_MIN, X_MAX)
                hijo[j*3+1] = np.clip(hijo[j*3+1], Y_MIN, Y_MAX)
                hijo[j*3+2] = np.clip(hijo[j*3+2], 0, R_MAX)
                
            nueva_poblacion[i] = hijo
            
        poblacion = nueva_poblacion # La nueva generación reemplaza a la vieja
    return mejor_individuo, mejor_fitness

# --- GREY WOLF OPTIMIZER (GWO) ---
def optimizacion_gwo(num_lobos, iteraciones):
    dim = N_MAX * 3  
    
    # Inicializar manada de lobos
    posiciones = np.random.uniform(low=0, high=100, size=(num_lobos, dim))
    for i in range(num_lobos): posiciones[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            
    # Inicializar jerarquía de líderes (empezamos con fitness negativo infinito)
    alfa_pos, alfa_score = np.zeros(dim), -float('inf')
    beta_pos, beta_score = np.zeros(dim), -float('inf')
    delta_pos, delta_score = np.zeros(dim), -float('inf')

    # Ciclo de cacería
    for it in range(iteraciones):
        # 1. Evaluar manada y actualizar líderes
        for i in range(num_lobos):
            # Reparar límites de posiciones
            for j in range(N_MAX):
                posiciones[i, j*3]   = np.clip(posiciones[i, j*3], X_MIN, X_MAX)
                posiciones[i, j*3+1] = np.clip(posiciones[i, j*3+1], Y_MIN, Y_MAX)
                posiciones[i, j*3+2] = np.clip(posiciones[i, j*3+2], 0, R_MAX)

            fit = calcular_fitness(posiciones[i])
            
            # Reasignación de jerarquía si encontramos mejores soluciones
            if fit > alfa_score:
                delta_score, delta_pos = beta_score, np.copy(beta_pos)
                beta_score, beta_pos = alfa_score, np.copy(alfa_pos)
                alfa_score, alfa_pos = fit, np.copy(posiciones[i])
            elif fit > beta_score:
                delta_score, delta_pos = beta_score, np.copy(beta_pos)
                beta_score, beta_pos = fit, np.copy(posiciones[i])
            elif fit > delta_score:
                delta_score, delta_pos = fit, np.copy(posiciones[i])

        # 'a' disminuye linealmente de 2 a 0. Controla el balance entre exploración y explotación.
        a = 2.0 - it * (2.0 / iteraciones)

        # 2. Actualizar posición de todos los lobos (Lobos Omega)
        for i in range(num_lobos):
            # Matrices aleatorias para los cálculos de vectores A y C
            r1, r2 = np.random.rand(3, dim), np.random.rand(3, dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            
            # Movimiento hacia el lobo Alfa
            D_alfa = np.abs(C[0] * alfa_pos - posiciones[i])
            X1 = alfa_pos - A[0] * D_alfa
            
            # Movimiento hacia el lobo Beta
            D_beta = np.abs(C[1] * beta_pos - posiciones[i])
            X2 = beta_pos - A[1] * D_beta
            
            # Movimiento hacia el lobo Delta
            D_delta = np.abs(C[2] * delta_pos - posiciones[i])
            X3 = delta_pos - A[2] * D_delta

            # La nueva posición del lobo es el promedio matemático de la influencia de los 3 líderes
            posiciones[i] = (X1 + X2 + X3) / 3.0
            
    return alfa_pos, alfa_score

# ==========================================
# 4. INTERFAZ DE CONSOLA Y REPORTES
# ==========================================
def limpiar_pantalla():
    # Detecta el sistema operativo para aplicar el comando correcto (cls en Windows, clear en Linux/Mac)
    os.system('cls' if os.name == 'nt' else 'clear')

# Función para formatear el vector ganador en una tabla legible
def imprimir_reporte(vector, fitness, tiempo, nombre_algoritmo):
    torres = np.reshape(vector, (N_MAX, 3))
    print(f"\n{'='*60}")
    print(f" RESULTADOS: {nombre_algoritmo}")
    print(f"{'='*60}")
    print(f"Fitness Final: {fitness:.4f}")
    print(f"Tiempo de ejecución: {tiempo:.2f} segundos")
    print(f"{'-'*60}")
    print(f"{'ID':<5} | {'Coord X':<10} | {'Coord Y':<10} | {'Radio (R)':<10} | {'Estado'}")
    print(f"{'-'*60}")
    
    activas = 0
    for idx, torre in enumerate(torres):
        x, y, r = torre
        # Si el algoritmo dejó el radio en menos de 0.5, consideramos que la torre no se construyó
        if r >= 0.5:
            print(f"T-{idx+1:<3} | {x:<10.2f} | {y:<10.2f} | {r:<10.2f} | ACTIVA")
            activas += 1
        else:
            print(f"T-{idx+1:<3} | {0:<10.2f} | {0:<10.2f} | {0:<10.2f} | APAGADA")
            
    print(f"{'-'*60}")
    print(f"Total torres construidas: {activas} de {N_MAX}\n")

# Lógica del menú principal interactivo
def menu_principal():
    pob_defecto = 30    # Tamaño de la población/enjambre/manada
    iter_defecto = 50   # Cantidad de iteraciones que durará la búsqueda
    
    while True: # Bucle infinito para mantener el programa abierto hasta que el usuario elija salir
        # limpiar_pantalla() # Descomenta esto si prefieres que se limpie la terminal
        print("\n" + "#"*50)
        print(" SISTEMA DE OPTIMIZACIÓN DE TORRES ".center(50, " "))
        print("#"*50)
        print("1. Ejecutar PSO (Enjambre de Partículas)")
        print("2. Ejecutar GA  (Algoritmo Genético)")
        print("3. Ejecutar GWO (Lobos Grises)")
        print("4. MODO COMPETENCIA (Ejecutar los 3 y comparar)")
        print("5. Salir")
        print("#"*50)
        
        opcion = input("Selecciona una opción (1-5): ")
        
        if opcion == '5':
            print("Saliendo del sistema...")
            break
            
        # Ejecución individual de algoritmos
        if opcion in ['1', '2', '3']:
            print(f"\nCalculando con {pob_defecto} individuos y {iter_defecto} iteraciones. Por favor espera...")
            inicio = time.time() # Iniciar cronómetro
            
            if opcion == '1':
                vec, fit = optimizacion_pso(pob_defecto, iter_defecto)
                nombre = "Particle Swarm Optimization (PSO)"
            elif opcion == '2':
                vec, fit = optimizacion_genetica(pob_defecto, iter_defecto)
                nombre = "Algoritmo Genético (GA)"
            elif opcion == '3':
                vec, fit = optimizacion_gwo(pob_defecto, iter_defecto)
                nombre = "Grey Wolf Optimizer (GWO)"
                
            tiempo = time.time() - inicio # Detener cronómetro
            imprimir_reporte(vec, fit, tiempo, nombre)
            input("Presiona ENTER para continuar...")
            
        # Ejecución en competencia (los 3 al mismo tiempo)
        elif opcion == '4':
            print(f"\nIniciando competencia ({pob_defecto} ind / {iter_defecto} iter)...")
            
            t_pso = time.time()
            vec_pso, fit_pso = optimizacion_pso(pob_defecto, iter_defecto)
            t_pso = time.time() - t_pso
            
            t_ga = time.time()
            vec_ga, fit_ga = optimizacion_genetica(pob_defecto, iter_defecto)
            t_ga = time.time() - t_ga
            
            t_gwo = time.time()
            vec_gwo, fit_gwo = optimizacion_gwo(pob_defecto, iter_defecto)
            t_gwo = time.time() - t_gwo
            
            print("\n" + "="*50)
            print(" TABLA DE POSICIONES FINAL ")
            print("="*50)
            resultados = [
                ("PSO", fit_pso, t_pso),
                ("GA", fit_ga, t_ga),
                ("GWO", fit_gwo, t_gwo)
            ]
            # Ordenar la lista basándose en el Fitness (índice 1), de mayor a menor (reverse=True)
            resultados.sort(key=lambda x: x[1], reverse=True)
            
            print(f"{'Puesto':<10} | {'Algoritmo':<10} | {'Fitness':<15} | {'Tiempo'}")
            print("-" * 50)
            # Imprimir el podio de ganadores
            for i, (alg, fit, t) in enumerate(resultados):
                print(f"{i+1:<10} | {alg:<10} | {fit:<15.4f} | {t:.2f} s")
            
            print("="*50)
            input("\nPresiona ENTER para volver al menú...")
        else:
            print("Opción no válida. Intenta de nuevo.")

# Punto de entrada estándar en Python. Si ejecutas este archivo directamente, se inicia el menú.
if __name__ == "__main__":
    menu_principal()











    # ==========================================
# PASO 2: EL "JUEZ" (FUNCIÓN OBJETIVO O FITNESS)
# ==========================================

def calcular_fitness(vector_individuo):
    # 1. TRADUCCIÓN: El PSO nos manda una lista plana de 30 números [x1, y1, r1, x2, y2, r2...]
    # Aquí la convertimos en una tabla ordenada de 10 filas (torres) y 3 columnas (x, y, r)
    torres = np.reshape(vector_individuo, (N_MAX, 3))
    
    # Contadores iniciales
    costo_total = 0
    penalizacion_frontera = 0
    
    # Nuestro "lienzo en blanco" de la ciudad. Una cuadrícula llena de ceros.
    # Aquí sumaremos qué zonas tienen cobertura.
    mapa_cobertura = np.zeros((RESOLUCION, RESOLUCION))
    
    # 2. EVALUACIÓN TORRE POR TORRE
    for torre in torres:
        x, y, r = torre  # Extraemos las coordenadas y el radio
        
        # Regla de negocio: Si el algoritmo propuso un radio menor a 0.5, 
        # significa que decidió NO construir esta torre para ahorrar dinero.
        if r < 0.5: 
            continue 
        
        # Límite técnico: Aseguramos que el radio no pase del máximo permitido
        r = min(r, R_MAX)
        
        # Fronteras: Si la torre está fuera del mapa, anotamos la multa y pasamos a la siguiente
        if not (X_MIN <= x <= X_MAX) or not (Y_MIN <= y <= Y_MAX):
            penalizacion_frontera += W_FUERA
            continue

        # Facturación: Sumamos el costo de esta antena (Área = Pi * radio al cuadrado)
        costo_total += np.pi * (r**2)
        
        # --- MAGIA VECTORIAL DE NUMPY ---
        # Pitágoras: Distancia desde la antena (x,y) a TODOS los 10,000 píxeles de la ciudad al mismo tiempo.
        distancia_cuadrada = (X_GRID - x)**2 + (Y_GRID - y)**2
        
        # Marcamos con "True" (1) los píxeles donde la señal sí llega (distancia <= radio al cuadrado)
        # y se lo sumamos a nuestro lienzo de cobertura.
        mapa_cobertura += (distancia_cuadrada <= (r**2))

    # 3. CÁLCULO DE ÁREAS FINALES
    # Cobertura: Contamos cuántos píxeles del mapa tienen al menos 1 señal llegando a ellos.
    area_cubierta = np.sum(mapa_cobertura >= 1) * AREA_POR_PUNTO
    
    # Solapamiento: Si a un píxel le llegan 3 señales, hay 2 de sobra. 
    # Le restamos 1 a todo el mapa, quitamos los negativos (con np.maximum) y sumamos el desperdicio.
    solapamiento = np.sum(np.maximum(0, mapa_cobertura - 1)) * AREA_POR_PUNTO

    # 4. LA CALIFICACIÓN FINAL (Ecuación matemática)
    # Queremos MAXIMIZAR esto: Multiplicamos el área por sus premios y le restamos los castigos.
    fitness = (W_COB * area_cubierta) - (W_COSTO * costo_total) - (W_SOLAP * solapamiento) - penalizacion_frontera
    
    return fitness



    # ==========================================
# PASO 3: EL ENJAMBRE (ALGORITMO PSO)
# ==========================================

def optimizacion_pso(num_particulas=30, iteraciones=50):
    # ¿Cuántos números tiene que mover cada partícula? 
    # 10 torres * 3 datos (X, Y, Radio) = 30 dimensiones.
    dim = N_MAX * 3  
    
    # --- LOS "CEREBROS" DE LOS PÁJAROS ---
    w = 0.7    # Inercia: Qué tan tercos son para seguir la dirección que ya traían (0 a 1)
    c1 = 1.5   # Cognitivo: Qué tanta importancia le dan a su PROPIA memoria
    c2 = 1.5   # Social: Qué tanta importancia le dan a lo que dice el LÍDER de la bandada
    
    # 1. EL BIG BANG (Generar el enjambre inicial al azar)
    # Creamos una matriz gigante donde cada fila es una partícula y cada columna es una coordenada o radio
    posiciones = np.random.uniform(low=0, high=100, size=(num_particulas, dim))
    
    # Ajuste fino: Los radios no pueden empezar en 100, deben empezar entre 0 y R_MAX
    # Usamos slicing de NumPy [i, 2::3] para modificar solo las columnas de los radios (índices 2, 5, 8...)
    for i in range(num_particulas): 
        posiciones[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            
    # Al principio nadie se mueve, velocidad = 0
    velocidades = np.zeros((num_particulas, dim))
    
    # 2. LAS MEMORIAS (Personal y Global)
    pbest_posiciones = np.copy(posiciones) # Memoria personal: "El mejor lugar en el que YO he estado"
    pbest_fitness = np.zeros(num_particulas)
    
    print("El juez está evaluando la posición inicial de la bandada...")
    # Calificamos a todos los pájaros por primera vez
    for i in range(num_particulas):
        pbest_fitness[i] = calcular_fitness(posiciones[i])
    
    # Buscamos al pájaro que tuvo más suerte al inicio y lo nombramos líder temporal
    gbest_idx = np.argmax(pbest_fitness) 
    gbest_posicion = np.copy(pbest_posiciones[gbest_idx]) # Memoria global: "El mejor lugar en el que TODOS hemos estado"
    gbest_fitness = pbest_fitness[gbest_idx]

    print("¡Empieza el vuelo (Optimizando)...!")
    
    # 3. EL BUCLE DE EVOLUCIÓN (El paso del tiempo)
    for it in range(iteraciones):
        # r1 y r2 son los "dados" del destino. Le dan un toque de caos e imprevisibilidad al vuelo
        r1 = np.random.rand(num_particulas, dim)
        r2 = np.random.rand(num_particulas, dim)
        
        # --- LA ECUACIÓN MÁGICA DE PSO ---
        # Nueva velocidad = (Inercia) + (Atracción a la memoria propia) + (Atracción al líder)
        velocidades = (w * velocidades) + \
                      (c1 * r1 * (pbest_posiciones - posiciones)) + \
                      (c2 * r2 * (gbest_posicion - posiciones))
        
        # Movemos las partículas sumándole la velocidad a su posición actual
        posiciones += velocidades  
        
        # 4. REVISIÓN Y REGLAS FÍSICAS
        for i in range(num_particulas):
            for j in range(N_MAX):
                # np.clip funciona como una "pared de cristal". Si un pájaro intenta volar 
                # fuera del mapa o hacer un radio infinito, lo forzamos a quedarse en el límite.
                posiciones[i, j*3]     = np.clip(posiciones[i, j*3], X_MIN, X_MAX)     # X
                posiciones[i, j*3+1]   = np.clip(posiciones[i, j*3+1], Y_MIN, Y_MAX)   # Y
                posiciones[i, j*3+2]   = np.clip(posiciones[i, j*3+2], 0, R_MAX)       # Radio
            
            # Mandamos el nuevo mapa al juez
            fit = calcular_fitness(posiciones[i])
            
            # 5. ACTUALIZAR MEMORIAS
            if fit > pbest_fitness[i]:
                # ¡El pájaro encontró un lugar mejor que su recuerdo anterior! Lo anota.
                pbest_fitness[i] = fit
                pbest_posiciones[i] = np.copy(posiciones[i])
                
                if fit > gbest_fitness:
                    # ¡Impresionante! Este pájaro superó al líder de la bandada. Ahora él es el nuevo rey.
                    gbest_fitness = fit
                    gbest_posicion = np.copy(posiciones[i])
                    
        # Imprimimos el progreso cada 10 iteraciones para no saturar la pantalla
        if (it + 1) % 10 == 0:
            print(f"Iteración {it+1}/{iteraciones} -> Mejor puntaje de la bandada: {gbest_fitness:.2f}")
                    
    # Al final de todas las iteraciones, devolvemos el mejor mapa que encontramos
    return gbest_posicion, gbest_fitness