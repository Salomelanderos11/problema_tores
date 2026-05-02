import numpy as np
import time

# -----------------------------------------
# Variables Globales y Parámetros
# -----------------------------------------

alfa = 0.6  # Importancia relativa de la latencia
beta = 0.4  # Importancia relativa del desbalance de carga

penalizacion_capacidad = 10000.0
penalizacion_umbral = 2000.0
penalizacion_demanda = 3000.0
penalizacion_conectividad = 5000.0

# Estructura de la red:
# Lista de tuplas -> [((Nodo_Origen, Nodo_Destino), Capacidad, Demanda, Latencia, Umbral_Congestion)]
enlaces_red = [
    (("N1", "N2"), 100, 50, 10, 0.75),
    (("N1", "N3"), 200, 30, 15, 0.85),
    (("N2", "N4"), 150, 60, 12, 0.80),
    (("N3", "N4"), 120, 40, 8,  0.75),
    (("N2", "N3"), 80,  20, 5,  0.90)
]

dim = len(enlaces_red)


# -----------------------------------------
# Función Fitness Modificada
# -----------------------------------------

def calcular_fitness(solucion_rutas):
    """
    Calcula la aptitud (fitness) de una configuración de red.
    
    Parámetros:
    solucion_rutas (arreglo/lista): Tráfico asignado a cada enlace de la red.
    """
    latencia_total = 0
    penalizacion = 0
    utilizaciones = []
    
    x = solucion_rutas
    
    # Conservación de flujo en nodos intermedios:
    # N2: Inflow (x[0]) == Outflow (x[2] + x[4])
    # N3: Inflow (x[1] + x[4]) == Outflow (x[3])
    n2_balance = abs(x[0] - (x[2] + x[4]))
    n3_balance = abs(x[1] + x[4] - x[3])
    
    penalizacion += penalizacion_conectividad * (n2_balance + n3_balance)
    
    for idx, enlace in enumerate(enlaces_red):
        (n1, n2), capacidad, demanda, latencia, umbral = enlace
        
        trafico_asignado = x[idx]
        
        # Restricción 1: Supera la capacidad máxima del enlace
        if trafico_asignado > capacidad:
            penalizacion += penalizacion_capacidad * (trafico_asignado - capacidad)
        elif trafico_asignado < 0:
            penalizacion += penalizacion_capacidad * abs(trafico_asignado)
            
        # Restricción 2: No cumple con la demanda requerida
        if trafico_asignado < demanda:
            penalizacion += penalizacion_demanda * (demanda - trafico_asignado)
            
        # Latencia Ponderada: Tráfico * Latencia
        latencia_total += latencia * trafico_asignado
        
        # Restricción 3: Supera el umbral de congestión
        utilizacion = (trafico_asignado / capacidad) if capacidad > 0 else 0
        utilizaciones.append(utilizacion)
        
        if utilizacion > umbral:
            penalizacion += penalizacion_umbral * (utilizacion - umbral)
            
    # Calcular el desbalance de la carga usando la desviación estándar
    desbalance = np.std(utilizaciones) if len(utilizaciones) > 0 else 0
    
    # Función Objetivo: Minimización (el valor más bajo es el mejor)
    fitness = (alfa * latencia_total) + (beta * desbalance) + penalizacion
    
    return fitness


# -----------------------------------------
# Población
# -----------------------------------------

def crear_poblacion(n):
    """
    Crea una población inicial para los algoritmos.
    """
    pob = np.zeros((n, dim))
    for i in range(n):
        for d in range(dim):
            capacidad = enlaces_red[d][1]
            pob[i, d] = np.random.uniform(0, capacidad)
            
    return pob


# -----------------------------------------
# 1. PSO (Particle Swarm Optimization)
# -----------------------------------------

def optimizacion_pso(n, iters):
    w, c1, c2 = 0.7, 1.5, 1.5
    pos = crear_poblacion(n)
    vel = np.zeros((n, dim))
    
    pbest = pos.copy()
    pfit = np.array([calcular_fitness(p) for p in pos])
    
    mejor_idx = np.argmin(pfit)
    gbest = pbest[mejor_idx].copy()
    gfit = pfit[mejor_idx]
    
    for _ in range(iters):
        for i in range(n):
            for d in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                
                vel[i, d] = (
                    w * vel[i, d]
                    + c1 * r1 * (pbest[i, d] - pos[i, d])
                    + c2 * r2 * (gbest[d] - pos[i, d])
                )
                
                pos[i, d] += vel[i, d]
                
            for d in range(dim):
                capacidad = enlaces_red[d][1]
                pos[i, d] = np.clip(pos[i, d], 0, capacidad)
                
            fit = calcular_fitness(pos[i])
            if fit < pfit[i]:
                pfit[i] = fit
                pbest[i] = pos[i].copy()
                if fit < gfit:
                    gfit = fit
                    gbest = pos[i].copy()
                    
    return gbest, gfit


# -----------------------------------------
# 2. GA (Algoritmo Genético)
# -----------------------------------------

def optimizacion_genetica(n, iters):
    pob = crear_poblacion(n)
    mejor = None
    mejor_fit = 1e15
    
    for _ in range(iters):
        fit = np.array([calcular_fitness(p) for p in pob])
        idx = np.argmin(fit)
        
        if fit[idx] < mejor_fit:
            mejor_fit = fit[idx]
            mejor = pob[idx].copy()
            
        nueva = np.zeros_like(pob)
        nueva[0] = mejor.copy()
        
        for i in range(1, n):
            p1, p2 = np.random.randint(0, n), np.random.randint(0, n)
            hijo = np.where(np.random.rand(dim) > 0.5, pob[p1], pob[p2])
            
            if np.random.rand() < 0.15:
                hijo += np.random.normal(0, 5, dim)
                
            for d in range(dim):
                capacidad = enlaces_red[d][1]
                hijo[d] = np.clip(hijo[d], 0, capacidad)
                
            nueva[i] = hijo
            
        pob = nueva
        
    return mejor, mejor_fit


# -----------------------------------------
# 3. GWO (Grey Wolf Optimizer)
# -----------------------------------------

def optimizacion_gwo(n, iters):
    pos = crear_poblacion(n)
    
    alfa_pos = pos[0].copy()
    alfa_fit = 1e15
    beta_pos = pos[1].copy()
    beta_fit = 1e15
    delta_pos = pos[2].copy()
    delta_fit = 1e15
    
    for i in range(n):
        fit = calcular_fitness(pos[i])
        if fit < alfa_fit:
            delta_fit = beta_fit
            delta_pos = beta_pos.copy()
            beta_fit = alfa_fit
            beta_pos = alfa_pos.copy()
            alfa_fit = fit
            alfa_pos = pos[i].copy()
        elif fit < beta_fit:
            delta_fit = beta_fit
            delta_pos = beta_pos.copy()
            beta_fit = fit
            beta_pos = pos[i].copy()
        elif fit < delta_fit:
            delta_fit = fit
            delta_pos = pos[i].copy()
            
    for it in range(iters):
        a = 2 - 2 * (it / iters)
        
        for i in range(n):
            for d in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alfa = abs(C1 * alfa_pos[d] - pos[i, d])
                X1 = alfa_pos[d] - A1 * D_alfa
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[d] - pos[i, d])
                X2 = beta_pos[d] - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[d] - pos[i, d])
                X3 = delta_pos[d] - A3 * D_delta
                
                pos[i, d] = (X1 + X2 + X3) / 3
                
            for d in range(dim):
                capacidad = enlaces_red[d][1]
                pos[i, d] = np.clip(pos[i, d], 0, capacidad)
                
            fit = calcular_fitness(pos[i])
            if fit < alfa_fit:
                delta_fit = beta_fit
                delta_pos = beta_pos.copy()
                beta_fit = alfa_fit
                beta_pos = alfa_pos.copy()
                alfa_fit = fit
                alfa_pos = pos[i].copy()
            elif fit < beta_fit:
                delta_fit = beta_fit
                delta_pos = beta_pos.copy()
                beta_fit = fit
                beta_pos = pos[i].copy()
            elif fit < delta_fit:
                delta_fit = fit
                delta_pos = pos[i].copy()
                
    return alfa_pos, alfa_fit


# -----------------------------------------
# 4. ABC (Artificial Bee Colony)
# -----------------------------------------

def optimizacion_abc(n, iters):
    pos = crear_poblacion(n)
    fit = np.array([calcular_fitness(p) for p in pos])
    intentos = np.zeros(n)
    limite = 20
    
    mejor_idx = np.argmin(fit)
    mejor = pos[mejor_idx].copy()
    mejor_fit = fit[mejor_idx]
    
    for _ in range(iters):
        # Abejas empleadas
        for i in range(n):
            k = np.random.randint(0, n)
            while k == i:
                k = np.random.randint(0, n)
                
            v = pos[i].copy()
            t = np.random.randint(dim)
            phi = np.random.uniform(-1, 1)
            v[t] = pos[i, t] + phi * (pos[i, t] - pos[k, t])
            
            capacidad = enlaces_red[t][1]
            v[t] = np.clip(v[t], 0, capacidad)
            
            f = calcular_fitness(v)
            if f < fit[i]:
                pos[i] = v
                fit[i] = f
                intentos[i] = 0
            else:
                intentos[i] += 1
                
        # Abejas observadoras
        inv_fit = 1.0 / (fit + 1e-10)
        prob = inv_fit / np.sum(inv_fit)
        
        for i in range(n):
            r = np.random.rand()
            cumulative_prob = 0
            target_idx = 0
            for j in range(n):
                cumulative_prob += prob[j]
                if r < cumulative_prob:
                    target_idx = j
                    break
                    
            k = np.random.randint(0, n)
            while k == target_idx:
                k = np.random.randint(0, n)
                
            v = pos[target_idx].copy()
            t = np.random.randint(dim)
            phi = np.random.uniform(-1, 1)
            v[t] = pos[target_idx, t] + phi * (pos[target_idx, t] - pos[k, t])
            
            capacidad = enlaces_red[t][1]
            v[t] = np.clip(v[t], 0, capacidad)
            
            f = calcular_fitness(v)
            if f < fit[target_idx]:
                pos[target_idx] = v
                fit[target_idx] = f
                intentos[target_idx] = 0
            else:
                intentos[target_idx] += 1
                
        # Abejas exploradoras
        for i in range(n):
            if intentos[i] > limite:
                for d in range(dim):
                    capacidad = enlaces_red[d][1]
                    pos[i, d] = np.random.uniform(0, capacidad)
                fit[i] = calcular_fitness(pos[i])
                intentos[i] = 0
                
        idx = np.argmin(fit)
        if fit[idx] < mejor_fit:
            mejor_fit = fit[idx]
            mejor = pos[idx].copy()
            
    return mejor, mejor_fit


# -----------------------------------------
# 5. AIS (Artificial Immune System)
# -----------------------------------------

def optimizacion_ais(n, iters):
    base = max(10, n // 3)
    anticuerpos = crear_poblacion(base)
    
    fit = np.array([calcular_fitness(a) for a in anticuerpos])
    idx = np.argsort(fit)
    anticuerpos = anticuerpos[idx]
    fit = fit[idx]
    
    mejor = anticuerpos[0].copy()
    mejor_fit = fit[0]
    
    for _ in range(iters):
        idx = np.argsort(fit)
        anticuerpos = anticuerpos[idx]
        fit = fit[idx]
        
        if fit[0] < mejor_fit:
            mejor_fit = fit[0]
            mejor = anticuerpos[0].copy()
            
        nuevos = []
        for i in range(base):
            tasa = (i + 1) / base
            for _ in range(3):
                clon = anticuerpos[i].copy()
                if np.random.rand() < 0.8:
                    clon += np.random.normal(0, 3 * tasa, dim)
                    
                for d in range(dim):
                    capacidad = enlaces_red[d][1]
                    clon[d] = np.clip(clon[d], 0, capacidad)
                    
                nuevos.append(clon)
                
        nuevos = np.array(nuevos)
        fit_n = np.array([calcular_fitness(c) for c in nuevos])
        mejores = np.argsort(fit_n)[:base]
        
        anticuerpos = nuevos[mejores]
        fit = fit_n[mejores]
        
    return mejor, mejor_fit


# -----------------------------------------
# Evaluación y Ejecución
# -----------------------------------------

def evaluar_algoritmo(func, nombre, reps, n, iters):
    resultados = []
    t_tot = 0
    
    print(f"\n--- Evaluando {nombre} (Múltiples corridas) ---")
    for i in range(reps):
        t0 = time.time()
        _, f = func(n, iters)
        t_tot += (time.time() - t0)
        resultados.append(f)
        print(f" Corrida {i+1} -> Fitness: {f:.4f}")
        
    resultados = np.array(resultados)
    print(f"\n Promedio Fitness: {np.mean(resultados):.4f}")
    print(f" Desviación Estándar: {np.std(resultados, ddof=1):.4f}")
    print(f" Tiempo promedio por corrida: {(t_tot / reps):.4f}s")
    
    return resultados


def main():
    algoritmos = {
        "1": (optimizacion_pso, "PSO"),
        "2": (optimizacion_genetica, "GA"),
        "3": (optimizacion_gwo, "GWO"),
        "4": (optimizacion_abc, "ABC"),
        "5": (optimizacion_ais, "AIS")
    }
    
    while True:
        print("\n--- SELECCIÓN DE ALGORITMO ---")
        print("1 - PSO")
        print("2 - GA")
        print("3 - GWO")
        print("4 - ABC")
        print("5 - AIS")
        print("6 - Salir")
        
        op = input("Seleccione una opción: ")
        
        if op == "6":
            print("Saliendo...")
            break
            
        if op not in algoritmos:
            print("Opción no válida.")
            continue
            
        func, nombre = algoritmos[op]
        
        modo = input("Seleccione modo: [1] Corrida única o [2] Estadístico: ")
        
        if modo not in ["1", "2"]:
            print("Modo no válido.")
            continue
            
        n = int(input("Tamaño de población: "))
        iters = int(input("Número de iteraciones: "))
        
        if modo == "1":
            t0 = time.time()
            solucion, mejor_fit = func(n, iters)
            t_total = time.time() - t0
            
            print(f"\n--- RESULTADOS {nombre} ---")
            print(f"Fitness: {mejor_fit:.4f}")
            print(f"Tiempo total: {t_total:.4f} segundos")
            print("Asignación de tráfico en enlaces:")
            
            for i, enlace in enumerate(enlaces_red):
                (n1, n2), capacidad, demanda, latencia, umbral = enlace
                print(f" - Enlace {n1}->{n2} | Capacidad: {capacidad} | Asignado: {solucion[i]:.2f}")
                
        elif modo == "2":
            reps = int(input("Número de corridas para estadística: "))
            evaluar_algoritmo(func, nombre, reps, n, iters)


if __name__ == "__main__":
    main()