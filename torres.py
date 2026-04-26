import numpy as np
import time
import os

# ==========================================
# 1. PARÁMETROS GLOBALES Y MALLA VIRTUAL
# ==========================================
X_MIN, X_MAX = 0, 100  
Y_MIN, Y_MAX = 0, 100  
N_MAX = 10             
R_MAX = 20             

W_COB = 1.0     
W_COSTO = 0.5   
W_SOLAP = 0.8   
W_FUERA = 1000  
MAX_PORCENTAJE_SOLAPAMIENTO = 15.0  # Límite estricto de solapamiento

# Malla virtual para el cálculo vectorial con NumPy
RESOLUCION = 100
x_line = np.linspace(X_MIN, X_MAX, RESOLUCION)
y_line = np.linspace(Y_MIN, Y_MAX, RESOLUCION)
X_GRID, Y_GRID = np.meshgrid(x_line, y_line)
AREA_POR_PUNTO = ((X_MAX - X_MIN) / RESOLUCION) * ((Y_MAX - Y_MIN) / RESOLUCION)

# ==========================================
# 2. FUNCIÓN OBJETIVO COMPARTIDA (FITNESS)
# ==========================================
def calcular_fitness(vector_individuo):
    torres = np.reshape(vector_individuo, (N_MAX, 3))
    costo_total = 0
    penalizacion_frontera = 0
    mapa_cobertura = np.zeros((RESOLUCION, RESOLUCION))
    
    for torre in torres:
        x, y, r = torre
        if r < 0.5: continue 
        r = min(r, R_MAX)
        
        if not (X_MIN <= x <= X_MAX) or not (Y_MIN <= y <= Y_MAX):
            penalizacion_frontera += W_FUERA
            continue

        costo_total += np.pi * (r**2)
        distancia_cuadrada = (X_GRID - x)**2 + (Y_GRID - y)**2
        mapa_cobertura += (distancia_cuadrada <= (r**2))

    area_cubierta = np.sum(mapa_cobertura >= 1) * AREA_POR_PUNTO
    solapamiento = np.sum(np.maximum(0, mapa_cobertura - 1)) * AREA_POR_PUNTO

    # Restricción estricta de porcentaje
    if area_cubierta > 0:
        porcentaje_solapamiento = (solapamiento / area_cubierta) * 100
        if porcentaje_solapamiento > MAX_PORCENTAJE_SOLAPAMIENTO:
            penalizacion_frontera += W_FUERA

    return (W_COB * area_cubierta) - (W_COSTO * costo_total) - (W_SOLAP * solapamiento) - penalizacion_frontera

# ==========================================
# 3. ALGORITMOS BIOINSPIRADOS
# ==========================================

# --- PARTICLE SWARM OPTIMIZATION (PSO) ---
def optimizacion_pso(num_particulas, iteraciones):
    dim = N_MAX * 3  
    w, c1, c2 = 0.7, 1.5, 1.5  
    
    posiciones = np.random.uniform(low=0, high=100, size=(num_particulas, dim))
    for i in range(num_particulas): posiciones[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            
    velocidades = np.zeros((num_particulas, dim))
    pbest_posiciones = np.copy(posiciones)
    pbest_fitness = np.array([calcular_fitness(p) for p in posiciones])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_posicion = np.copy(pbest_posiciones[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    for it in range(iteraciones):
        r1, r2 = np.random.rand(num_particulas, dim), np.random.rand(num_particulas, dim)
        velocidades = w * velocidades + c1 * r1 * (pbest_posiciones - posiciones) + c2 * r2 * (gbest_posicion - posiciones)
        posiciones += velocidades
        
        for i in range(num_particulas):
            for j in range(N_MAX):
                posiciones[i, j*3]     = np.clip(posiciones[i, j*3], X_MIN, X_MAX)
                posiciones[i, j*3+1]   = np.clip(posiciones[i, j*3+1], Y_MIN, Y_MAX)
                posiciones[i, j*3+2]   = np.clip(posiciones[i, j*3+2], 0, R_MAX)
            
            fit = calcular_fitness(posiciones[i])
            if fit > pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_posiciones[i] = np.copy(posiciones[i])
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_posicion = np.copy(posiciones[i])
    return gbest_posicion, gbest_fitness

# --- ALGORITMO GENÉTICO (GA) ---
def optimizacion_genetica(tam_poblacion, generaciones):
    dim = N_MAX * 3  
    poblacion = np.random.uniform(low=0, high=100, size=(tam_poblacion, dim))
    for i in range(tam_poblacion): poblacion[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            
    mejor_individuo, mejor_fitness = None, -float('inf')

    for gen in range(generaciones):
        fitness_pob = np.array([calcular_fitness(ind) for ind in poblacion])
        
        idx_mejor = np.argmax(fitness_pob)
        if fitness_pob[idx_mejor] > mejor_fitness:
            mejor_fitness = fitness_pob[idx_mejor]
            mejor_individuo = np.copy(poblacion[idx_mejor])
            
        nueva_poblacion = np.zeros_like(poblacion)
        nueva_poblacion[0] = np.copy(mejor_individuo) # Elitismo
        
        for i in range(1, tam_poblacion):
            p1 = poblacion[np.random.choice(tam_poblacion, 3)[np.argmax(fitness_pob[np.random.choice(tam_poblacion, 3)])]]
            p2 = poblacion[np.random.choice(tam_poblacion, 3)[np.argmax(fitness_pob[np.random.choice(tam_poblacion, 3)])]]
            
            hijo = np.where(np.random.rand(dim) > 0.5, p1, p2)
            
            if np.random.rand() < 0.15: # Mutación
                hijo[0::3] += np.random.normal(0, 5, N_MAX)
                hijo[1::3] += np.random.normal(0, 5, N_MAX)
                hijo[2::3] += np.random.normal(0, 2, N_MAX)
            
            for j in range(N_MAX):
                hijo[j*3]   = np.clip(hijo[j*3], X_MIN, X_MAX)
                hijo[j*3+1] = np.clip(hijo[j*3+1], Y_MIN, Y_MAX)
                hijo[j*3+2] = np.clip(hijo[j*3+2], 0, R_MAX)
                
            nueva_poblacion[i] = hijo
        poblacion = nueva_poblacion
    return mejor_individuo, mejor_fitness

# --- GREY WOLF OPTIMIZER (GWO) ---
def optimizacion_gwo(num_lobos, iteraciones):
    dim = N_MAX * 3  
    posiciones = np.random.uniform(low=0, high=100, size=(num_lobos, dim))
    for i in range(num_lobos): posiciones[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            
    alfa_pos, alfa_score = np.zeros(dim), -float('inf')
    beta_pos, beta_score = np.zeros(dim), -float('inf')
    delta_pos, delta_score = np.zeros(dim), -float('inf')

    for it in range(iteraciones):
        for i in range(num_lobos):
            for j in range(N_MAX):
                posiciones[i, j*3]   = np.clip(posiciones[i, j*3], X_MIN, X_MAX)
                posiciones[i, j*3+1] = np.clip(posiciones[i, j*3+1], Y_MIN, Y_MAX)
                posiciones[i, j*3+2] = np.clip(posiciones[i, j*3+2], 0, R_MAX)

            fit = calcular_fitness(posiciones[i])
            
            if fit > alfa_score:
                delta_score, delta_pos = beta_score, np.copy(beta_pos)
                beta_score, beta_pos = alfa_score, np.copy(alfa_pos)
                alfa_score, alfa_pos = fit, np.copy(posiciones[i])
            elif fit > beta_score:
                delta_score, delta_pos = beta_score, np.copy(beta_pos)
                beta_score, beta_pos = fit, np.copy(posiciones[i])
            elif fit > delta_score:
                delta_score, delta_pos = fit, np.copy(posiciones[i])

        a = 2.0 - it * (2.0 / iteraciones)

        for i in range(num_lobos):
            r1, r2 = np.random.rand(3, dim), np.random.rand(3, dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            
            D_alfa = np.abs(C[0] * alfa_pos - posiciones[i])
            X1 = alfa_pos - A[0] * D_alfa
            D_beta = np.abs(C[1] * beta_pos - posiciones[i])
            X2 = beta_pos - A[1] * D_beta
            D_delta = np.abs(C[2] * delta_pos - posiciones[i])
            X3 = delta_pos - A[2] * D_delta

            posiciones[i] = (X1 + X2 + X3) / 3.0
            
    return alfa_pos, alfa_score

# --- ARTIFICIAL BEE COLONY (ABC) ---
def optimizacion_abc(tam_pob, iteraciones):
    dim = N_MAX * 3
    limite_estancamiento = 20 # Si una fuente de néctar no mejora en 20 turnos, se abandona.
    
    posiciones = np.random.uniform(low=0, high=100, size=(tam_pob, dim))
    for i in range(tam_pob): posiciones[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
    
    fitness = np.array([calcular_fitness(p) for p in posiciones])
    intentos = np.zeros(tam_pob) # Contador de estancamiento
    
    mejor_posicion = np.copy(posiciones[np.argmax(fitness)])
    mejor_fitness = np.max(fitness)

    for it in range(iteraciones):
        # 1. Abejas Empleadas: Modifican levemente la solución actual
        for i in range(tam_pob):
            vecino_k = np.random.choice([x for x in range(tam_pob) if x != i])
            v = np.copy(posiciones[i])
            # Seleccionamos una torre al azar para modificar (X, Y y Radio)
            torre_t = np.random.randint(N_MAX)
            phi = np.random.uniform(-1, 1, 3) 
            
            # Ecuación de movimiento ABC
            v[torre_t*3 : torre_t*3+3] += phi * (posiciones[i, torre_t*3 : torre_t*3+3] - posiciones[vecino_k, torre_t*3 : torre_t*3+3])
            
            # Reparar límites
            v[torre_t*3]   = np.clip(v[torre_t*3], X_MIN, X_MAX)
            v[torre_t*3+1] = np.clip(v[torre_t*3+1], Y_MIN, Y_MAX)
            v[torre_t*3+2] = np.clip(v[torre_t*3+2], 0, R_MAX)
            
            fit_v = calcular_fitness(v)
            if fit_v > fitness[i]:
                posiciones[i] = v
                fitness[i] = fit_v
                intentos[i] = 0 # Resetea el estancamiento
            else:
                intentos[i] += 1
                
        # 2. Abejas Observadoras: Van a las mejores soluciones
        # Convertimos fitness en probabilidades (Ruleta)
        fit_norm = fitness - np.min(fitness) + 1e-6 
        probabilidades = fit_norm / np.sum(fit_norm)
        
        for _ in range(tam_pob):
            i = np.random.choice(tam_pob, p=probabilidades)
            vecino_k = np.random.choice([x for x in range(tam_pob) if x != i])
            v = np.copy(posiciones[i])
            torre_t = np.random.randint(N_MAX)
            phi = np.random.uniform(-1, 1, 3)
            
            v[torre_t*3 : torre_t*3+3] += phi * (posiciones[i, torre_t*3 : torre_t*3+3] - posiciones[vecino_k, torre_t*3 : torre_t*3+3])
            
            v[torre_t*3]   = np.clip(v[torre_t*3], X_MIN, X_MAX)
            v[torre_t*3+1] = np.clip(v[torre_t*3+1], Y_MIN, Y_MAX)
            v[torre_t*3+2] = np.clip(v[torre_t*3+2], 0, R_MAX)
            
            fit_v = calcular_fitness(v)
            if fit_v > fitness[i]:
                posiciones[i] = v
                fitness[i] = fit_v
                intentos[i] = 0
            else:
                intentos[i] += 1

        # Actualizar la mejor solución global
        if np.max(fitness) > mejor_fitness:
            mejor_fitness = np.max(fitness)
            mejor_posicion = np.copy(posiciones[np.argmax(fitness)])

        # 3. Abejas Exploradoras: Si una solución se estanca, generar una nueva al azar
        for i in range(tam_pob):
            if intentos[i] > limite_estancamiento:
                posiciones[i] = np.random.uniform(0, 100, dim)
                posiciones[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
                fitness[i] = calcular_fitness(posiciones[i])
                intentos[i] = 0

    return mejor_posicion, mejor_fitness

# --- ARTIFICIAL IMMUNE SYSTEM (AIS) ---
def optimizacion_ais(tam_pob, iteraciones):
    dim = N_MAX * 3
    # Reducimos la población inicial para compensar la explosión de clones y mantener tiempos justos
    pob_base = max(10, tam_pob // 3) 
    num_clones = 3
    
    anticuerpos = np.random.uniform(low=0, high=100, size=(pob_base, dim))
    for i in range(pob_base): anticuerpos[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
    
    mejor_anticuerpo = None
    mejor_afinidad = -float('inf')

    for it in range(iteraciones):
        afinidades = np.array([calcular_fitness(a) for a in anticuerpos])
        
        # Ordenar anticuerpos de mejor a peor (Selección)
        indices_ordenados = np.argsort(afinidades)[::-1]
        anticuerpos = anticuerpos[indices_ordenados]
        afinidades = afinidades[indices_ordenados]
        
        if afinidades[0] > mejor_afinidad:
            mejor_afinidad = afinidades[0]
            mejor_anticuerpo = np.copy(anticuerpos[0])
            
        nuevos_anticuerpos = []
        
        # Clonación e Hipermutación
        for i in range(pob_base):
            # Tasa de mutación inversamente proporcional al rango. El mejor muta poco, el peor muta mucho.
            tasa_mutacion = (i + 1) / pob_base 
            
            for _ in range(num_clones):
                clon = np.copy(anticuerpos[i])
                
                # Mutación Gaussiana
                if np.random.rand() < 0.8: # 80% de probabilidad de que el clon mute
                    clon[0::3] += np.random.normal(0, 10 * tasa_mutacion, N_MAX) # Mutar X
                    clon[1::3] += np.random.normal(0, 10 * tasa_mutacion, N_MAX) # Mutar Y
                    clon[2::3] += np.random.normal(0, 3 * tasa_mutacion, N_MAX)  # Mutar Radio
                    
                # Reparar fenotipo
                for j in range(N_MAX):
                    clon[j*3]   = np.clip(clon[j*3], X_MIN, X_MAX)
                    clon[j*3+1] = np.clip(clon[j*3+1], Y_MIN, Y_MAX)
                    clon[j*3+2] = np.clip(clon[j*3+2], 0, R_MAX)
                    
                nuevos_anticuerpos.append(clon)
                
        # Evaluar todos los clones y seleccionar los mejores 'pob_base' para sobrevivir
        afinidades_clones = np.array([calcular_fitness(c) for c in nuevos_anticuerpos])
        mejores_indices = np.argsort(afinidades_clones)[::-1][:pob_base]
        
        anticuerpos = np.array(nuevos_anticuerpos)[mejores_indices]
        
        # Edición del receptor: Matar al 20% de los peores anticuerpos y reemplazarlos con aleatorios nuevos
        num_reemplazos = int(pob_base * 0.2)
        if num_reemplazos > 0:
            nuevos_aleatorios = np.random.uniform(0, 100, (num_reemplazos, dim))
            for i in range(num_reemplazos): nuevos_aleatorios[i, 2::3] = np.random.uniform(0, R_MAX, N_MAX)
            anticuerpos[-num_reemplazos:] = nuevos_aleatorios

    return mejor_anticuerpo, mejor_afinidad

# ==========================================
# 4. INTERFAZ DE CONSOLA Y REPORTES
# ==========================================
def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

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
        if r >= 0.5:
            print(f"T-{idx+1:<3} | {x:<10.2f} | {y:<10.2f} | {r:<10.2f} | ACTIVA")
            activas += 1
        else:
            print(f"T-{idx+1:<3} | {0:<10.2f} | {0:<10.2f} | {0:<10.2f} | APAGADA")
            
    print(f"{'-'*60}")
    print(f"Total torres construidas: {activas} de {N_MAX}\n")

def menu_principal():
    pob_defecto = 30
    iter_defecto = 50
    
    while True:
        print("\n" + "#"*55)
        print(" SISTEMA DE OPTIMIZACIÓN BIOINSPIRADA DE TORRES ".center(55, " "))
        print("#"*55)
        print("1. Ejecutar PSO (Enjambre de Partículas)")
        print("2. Ejecutar GA  (Algoritmo Genético)")
        print("3. Ejecutar GWO (Lobos Grises)")
        print("4. Ejecutar ABC (Colonia de Abejas)")
        print("5. Ejecutar AIS (Sistema Inmune Artificial)")
        print("6. MODO COMPETENCIA (Ejecutar los 5 y comparar)")
        print("7. Salir")
        print("#"*55)
        
        opcion = input("Selecciona una opción (1-7): ")
        
        if opcion == '7':
            print("Saliendo del sistema...")
            break
            
        if opcion in ['1', '2', '3', '4', '5']:
            print(f"\nCalculando con {pob_defecto} individuos y {iter_defecto} iteraciones. Por favor espera...")
            inicio = time.time()
            
            if opcion == '1':
                vec, fit = optimizacion_pso(pob_defecto, iter_defecto)
                nombre = "Particle Swarm Optimization (PSO)"
            elif opcion == '2':
                vec, fit = optimizacion_genetica(pob_defecto, iter_defecto)
                nombre = "Algoritmo Genético (GA)"
            elif opcion == '3':
                vec, fit = optimizacion_gwo(pob_defecto, iter_defecto)
                nombre = "Grey Wolf Optimizer (GWO)"
            elif opcion == '4':
                vec, fit = optimizacion_abc(pob_defecto, iter_defecto)
                nombre = "Artificial Bee Colony (ABC)"
            elif opcion == '5':
                vec, fit = optimizacion_ais(pob_defecto, iter_defecto)
                nombre = "Artificial Immune System (AIS)"
                
            tiempo = time.time() - inicio
            imprimir_reporte(vec, fit, tiempo, nombre)
            input("Presiona ENTER para continuar...")
            
        elif opcion == '6':
            print(f"\nIniciando gran competencia ({pob_defecto} ind / {iter_defecto} iter)...")
            
            resultados = []
            algoritmos = [
                ("PSO", optimizacion_pso),
                ("GA", optimizacion_genetica),
                ("GWO", optimizacion_gwo),
                ("ABC", optimizacion_abc),
                ("AIS", optimizacion_ais)
            ]
            
            for nombre, funcion in algoritmos:
                t_inicio = time.time()
                vec, fit = funcion(pob_defecto, iter_defecto)
                t_fin = time.time() - t_inicio
                resultados.append((nombre, fit, t_fin))
                print(f"[{nombre}] finalizado en {t_fin:.2f}s")
            
            print("\n" + "="*50)
            print(" TABLA DE POSICIONES FINAL ")
            print("="*50)
            
            # Ordenar de mayor a menor fitness
            resultados.sort(key=lambda x: x[1], reverse=True)
            
            print(f"{'Puesto':<10} | {'Algoritmo':<10} | {'Fitness':<15} | {'Tiempo'}")
            print("-" * 50)
            for i, (alg, fit, t) in enumerate(resultados):
                print(f"{i+1:<10} | {alg:<10} | {fit:<15.4f} | {t:.2f} s")
            
            print("="*50)
            input("\nPresiona ENTER para volver al menú...")
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    menu_principal()