import numpy as np
import time

# -----------------------------------------
# PARÁMETROS
# -----------------------------------------

alfa = 0.6
beta = 0.4

penalizacion_capacidad = 10000
penalizacion_umbral = 2000
penalizacion_demanda = 3000

# -----------------------------------------
# RED
# -----------------------------------------

enlaces = {
    # Conexiones desde N1
    ("N1", "N2"): (100, 10, 0.75),
    ("N1", "N3"): (200, 15, 0.85),
    ("N1", "N4"): (150, 12, 0.80),
    ("N1", "N5"): (80, 8, 0.70),
    ("N1", "N6"): (120, 14, 0.78),
    ("N1", "N7"): (90, 9, 0.82),
    ("N1", "N8"): (110, 11, 0.75),
    ("N1", "N9"): (160, 18, 0.88),
    ("N1", "N10"): (140, 13, 0.77),
    ("N1", "N11"): (130, 10, 0.81),

    # Conexiones desde N2
    ("N2", "N3"): (90, 5, 0.90),
    ("N2", "N4"): (200, 8, 0.85),
    ("N2", "N5"): (110, 16, 0.79),
    ("N2", "N6"): (150, 10, 0.75),
    ("N2", "N7"): (120, 12, 0.82),
    ("N2", "N8"): (80, 14, 0.77),
    ("N2", "N9"): (100, 7, 0.90),
    ("N2", "N10"): (160, 9, 0.83),
    ("N2", "N11"): (140, 11, 0.78),

    # Conexiones desde N3
    ("N3", "N4"): (120, 8, 0.75),
    ("N3", "N5"): (150, 11, 0.80),
    ("N3", "N6"): (100, 15, 0.73),
    ("N3", "N7"): (130, 7, 0.88),
    ("N3", "N8"): (160, 10, 0.77),
    ("N3", "N9"): (110, 13, 0.82),
    ("N3", "N10"): (90, 9, 0.79),
    ("N3", "N11"): (200, 6, 0.86),

    # Conexiones desde N4
    ("N4", "N5"): (100, 10, 0.82),
    ("N4", "N6"): (140, 7, 0.75),
    ("N4", "N7"): (110, 14, 0.79),
    ("N4", "N8"): (120, 12, 0.88),
    ("N4", "N9"): (80, 9, 0.74),
    ("N4", "N10"): (150, 11, 0.81),
    ("N4", "N11"): (90, 13, 0.77),

    # Conexiones desde N5
    ("N5", "N6"): (130, 8, 0.76),
    ("N5", "N7"): (160, 13, 0.84),
    ("N5", "N8"): (90, 9, 0.72),
    ("N5", "N9"): (140, 5, 0.89),
    ("N5", "N10"): (110, 12, 0.77),
    ("N5", "N11"): (120, 15, 0.80),

    # Conexiones desde N6
    ("N6", "N7"): (100, 6, 0.81),
    ("N6", "N8"): (110, 11, 0.74),
    ("N6", "N9"): (150, 14, 0.83),
    ("N6", "N10"): (80, 8, 0.78),
    ("N6", "N11"): (140, 10, 0.86),

    # Conexiones desde N7
    ("N7", "N8"): (120, 9, 0.75),
    ("N7", "N9"): (100, 12, 0.80),
    ("N7", "N10"): (130, 7, 0.77),
    ("N7", "N11"): (110, 10, 0.85)
}

# Demanda origen-destino
demandas = [
    ("N1","N4", 100)
]

# Rutas posibles
rutas = {
    ("N1","N4"): [
        [("N1","N2"), ("N2","N4")],
        [("N1","N3"), ("N3","N4")],
        [("N1","N2"), ("N2","N3"), ("N3","N4")]
    ]
}

lista_rutas = []
for od in rutas:
    for r in rutas[od]:
        lista_rutas.append((od, r))

dim = len(lista_rutas)

# -----------------------------------------
# FITNESS
# -----------------------------------------

def calcular_fitness(x):
    flujo_enlace = {e:0 for e in enlaces}

    # Construir flujo por enlace
    for i, ((o,d), ruta) in enumerate(lista_rutas):
        for e in ruta:
            flujo_enlace[e] += x[i]

    latencia_total = 0
    penalizacion = 0
    utilizaciones = []

    # Restricción demanda
    for (o,d,dem) in demandas:
        indices = [i for i,(od,_) in enumerate(lista_rutas) if od==(o,d)]
        flujo_total = sum(x[i] for i in indices)

        if flujo_total < dem:
            penalizacion += penalizacion_demanda * (dem - flujo_total)

    # Evaluar enlaces
    for e in enlaces:
        capacidad, latencia, umbral = enlaces[e]
        flujo = flujo_enlace[e]

        if flujo > capacidad:
            penalizacion += penalizacion_capacidad * (flujo - capacidad)

        latencia_total += latencia * flujo

        u = flujo / capacidad if capacidad>0 else 0
        utilizaciones.append(u)

        if u > umbral:
            penalizacion += penalizacion_umbral * (u - umbral)

    desbalance = np.std(utilizaciones) if len(utilizaciones) > 0 else 0

    return alfa*latencia_total + beta*desbalance + penalizacion

# -----------------------------------------
# POBLACIÓN
# -----------------------------------------

def crear_poblacion(n):
    return np.random.uniform(0, 100, (n, dim))

def clip(x):
    return np.clip(x, 0, 200)

# -----------------------------------------
# ALGORITMOS
# -----------------------------------------

def PSO(n, iters):
    w,c1,c2 = 0.7,1.5,1.5

    pos = crear_poblacion(n)
    vel = np.zeros_like(pos)

    pbest = pos.copy()
    pfit = np.array([calcular_fitness(p) for p in pos])

    gbest = pbest[np.argmin(pfit)].copy()
    gfit = np.min(pfit)

    for _ in range(iters):
        for i in range(n):
            r1,r2 = np.random.rand(), np.random.rand()

            vel[i] = (w*vel[i] +
                      c1*r1*(pbest[i]-pos[i]) +
                      c2*r2*(gbest-pos[i]))

            pos[i] = clip(pos[i] + vel[i])

            fit = calcular_fitness(pos[i])

            if fit < pfit[i]:
                pfit[i] = fit
                pbest[i] = pos[i].copy()

                if fit < gfit:
                    gfit = fit
                    gbest = pos[i].copy()

    return gbest, gfit


def GA(n,iters):
    pob = crear_poblacion(n)
    best, best_fit = None, 1e15

    for _ in range(iters):
        fit = np.array([calcular_fitness(p) for p in pob])

        idx = np.argmin(fit)
        if fit[idx] < best_fit:
            best_fit = fit[idx]
            best = pob[idx].copy()

        nueva = [best.copy()]

        for _ in range(n-1):
            p1,p2 = pob[np.random.randint(n)], pob[np.random.randint(n)]
            hijo = np.where(np.random.rand(dim)>0.5, p1, p2)

            if np.random.rand()<0.2:
                hijo += np.random.normal(0,5,dim)

            nueva.append(clip(hijo))

        pob = np.array(nueva)

    return best, best_fit


def GWO(n,iters):
    pos = crear_poblacion(n)

    for _ in range(iters):
        fit = np.array([calcular_fitness(p) for p in pos])
        idx = np.argsort(fit)

        alfa = pos[idx[0]]
        beta = pos[idx[1]]
        delta = pos[idx[2]]

        a = 2*(1 - _/iters)

        for i in range(n):
            r1,r2 = np.random.rand(),np.random.rand()
            A1 = 2*a*r1-a; C1 = 2*r2
            D1 = abs(C1*alfa-pos[i])
            X1 = alfa-A1*D1

            r1,r2 = np.random.rand(),np.random.rand()
            A2 = 2*a*r1-a; C2 = 2*r2
            D2 = abs(C2*beta-pos[i])
            X2 = beta-A2*D2

            r1,r2 = np.random.rand(),np.random.rand()
            A3 = 2*a*r1-a; C3 = 2*r2
            D3 = abs(C3*delta-pos[i])
            X3 = delta-A3*D3

            pos[i] = clip((X1+X2+X3)/3)

    fit = np.array([calcular_fitness(p) for p in pos])
    i = np.argmin(fit)

    return pos[i], fit[i]


def ABC(n,iters):
    pos = crear_poblacion(n)
    fit = np.array([calcular_fitness(p) for p in pos])

    best = pos[np.argmin(fit)].copy()
    best_fit = np.min(fit)

    for _ in range(iters):

        for i in range(n):
            k = np.random.randint(n)
            v = pos[i].copy()
            d = np.random.randint(dim)

            v[d] += np.random.uniform(-1,1)*(v[d]-pos[k,d])
            v = clip(v)

            f = calcular_fitness(v)
            if f < fit[i]:
                pos[i],fit[i] = v,f

        for i in range(n):
            if np.random.rand()<0.5:
                k = np.random.randint(n)
                v = pos[i] + np.random.uniform(-1,1)*(pos[i]-pos[k])
                v = clip(v)

                f = calcular_fitness(v)
                if f < fit[i]:
                    pos[i],fit[i] = v,f

        i = np.argmin(fit)
        if fit[i] < best_fit:
            best_fit = fit[i]
            best = pos[i].copy()

    return best,best_fit


def AIS(n,iters):
    pob = crear_poblacion(n)

    for _ in range(iters):
        fit = np.array([calcular_fitness(p) for p in pob])
        idx = np.argsort(fit)

        pob = pob[idx[:n//2]]

        clones = []
        for p in pob:
            for _ in range(3):
                c = p + np.random.normal(0,3,dim)
                clones.append(clip(c))

        pob = np.array(clones)

    fit = np.array([calcular_fitness(p) for p in pob])
    i = np.argmin(fit)

    return pob[i], fit[i]

# -----------------------------------------
# MENÚ INTERACTIVO
# -----------------------------------------

def main():
    algs = {
        "1": (PSO, "PSO"),
        "2": (GA, "GA"),
        "3": (GWO, "GWO"),
        "4": (ABC, "ABC"),
        "5": (AIS, "AIS")
    }

    while True:
        print("\n" + "="*40)
        print("  SISTEMA DE OPTIMIZACIÓN DE RED DE DATOS")
        print("="*40)
        print("1 - Optimización por Enjambre de Partículas (PSO)")
        print("2 - Algoritmo Genético (GA)")
        print("3 - Grey Wolf Optimizer (GWO)")
        print("4 - Colonia de Abejas Artificiales (ABC)")
        print("5 - Sistema Inmunitario Artificial (AIS)")
        print("6 - Salir")
        
        op = input("\nSeleccione un algoritmo: ")

        if op == "6":
            print("\n¡Gracias por utilizar el sistema! Saliendo...")
            break

        if op not in algs:
            print("\nOpción inválida, por favor intente de nuevo.")
            continue

        func, nombre = algs[op]

        modo = input("Seleccione modo: [1] Corrida única o [2] Estadístico (múltiples corridas): ")

        if modo not in ["1", "2"]:
            print("\nModo inválido.")
            continue

        try:
            n = int(input("Tamaño de población: "))
            iters = int(input("Número de iteraciones: "))
        except ValueError:
            print("\nPor favor, introduzca un número entero válido.")
            continue

        if modo == "1":
            t0 = time.time()
            sol, fit = func(n, iters)
            t = time.time() - t0

            print(f"\n--- RESULTADOS {nombre} ---")
            print(f"Fitness óptimo: {fit:.4f}")
            print(f"Tiempo empleado: {t:.4f} segundos")
            print("Asignación de flujo en las rutas:")
            
            for i, ((o, d), ruta) in enumerate(lista_rutas):
                print(f" * {o}->{d} | Ruta: {ruta} | Flujo asignado: {sol[i]:.2f}")

        elif modo == "2":
            reps = int(input("Número de corridas para estadística: "))
            resultados = []
            t_tot = 0

            print(f"\n--- Evaluando {nombre} ({reps} corridas) ---")
            for i in range(reps):
                t0 = time.time()
                _, fit = func(n, iters)
                t_tot += (time.time() - t0)
                resultados.append(fit)
                print(f"  Corrida {i+1} -> Fitness: {fit:.4f}")

            resultados = np.array(resultados)
            print("\n--- RESUMEN ESTADÍSTICO ---")
            print(f"  Fitness Promedio: {np.mean(resultados):.4f}")
            print(f"  Desviación Estándar: {np.std(resultados, ddof=1):.4f}")
            print(f"  Tiempo promedio por corrida: {t_tot / reps:.4f} s")


if __name__ == "__main__":
    main()