import numpy as np
import time
import os


# -----------------------------------------
# parametros generales
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


# malla del terreno
linea_x = np.linspace(x_min, x_max, resolucion)
linea_y = np.linspace(y_min, y_max, resolucion)

grid_x, grid_y = np.meshgrid(linea_x, linea_y)

area_punto = ((x_max-x_min)/resolucion) * ((y_max-y_min)/resolucion)


# -----------------------------------------
# funcion fitness
# -----------------------------------------

def calcular_fitness(individuo):

    torres = np.reshape(individuo,(max_torres,3))

    costo_total = 0
    penalizacion = 0

    mapa = np.zeros((resolucion,resolucion))


    for torre in torres:

        x = torre[0]
        y = torre[1]
        r = torre[2]


        if r < 0.5:
            continue


        if r > radio_max:
            r = radio_max


        if x < x_min or x > x_max:
            penalizacion = penalizacion + penalizacion_fuera
            continue

        if y < y_min or y > y_max:
            penalizacion = penalizacion + penalizacion_fuera
            continue


        costo_total = costo_total + np.pi*(r*r)


        distancia2 = (grid_x-x)**2 + (grid_y-y)**2

        cobertura = distancia2 <= r*r

        mapa = mapa + cobertura



    area_cubierta = np.sum(mapa>=1) * area_punto

    solapamiento = np.sum(
        np.maximum(0,mapa-1)
    ) * area_punto



    if area_cubierta > 0:

        porcentaje = (solapamiento/area_cubierta)*100

        if porcentaje > max_solapamiento:
            penalizacion = penalizacion + penalizacion_fuera



    fitness = (
        peso_cobertura*area_cubierta
        -
        peso_costo*costo_total
        -
        peso_solapamiento*solapamiento
        -
        penalizacion
    )

    return fitness



# -----------------------------------------
# crear poblacion inicial
# -----------------------------------------

def crear_poblacion(tamano):

    dimension = max_torres*3

    poblacion = np.random.uniform(
        0,
        100,
        (tamano,dimension)
    )

    for i in range(tamano):

        for j in range(max_torres):

            poblacion[i,j*3+2] = np.random.uniform(
                0,
                radio_max
            )

    return poblacion



# -----------------------------------------
# algoritmo pso
# -----------------------------------------

def optimizacion_pso(particulas,iteraciones):

    dimension = max_torres*3

    w = 0.7
    c1 = 1.5
    c2 = 1.5


    posiciones = crear_poblacion(particulas)

    velocidades = np.zeros(
        (particulas,dimension)
    )


    mejor_local = posiciones.copy()

    fitness_local = np.zeros(particulas)

    for i in range(particulas):
        fitness_local[i] = calcular_fitness(
            posiciones[i]
        )


    indice = np.argmax(fitness_local)

    mejor_global = posiciones[indice].copy()

    fitness_global = fitness_local[indice]



    for it in range(iteraciones):

        for i in range(particulas):

            for d in range(dimension):

                r1 = np.random.rand()
                r2 = np.random.rand()

                velocidades[i,d] = (
                    w*velocidades[i,d]
                    +
                    c1*r1*(
                      mejor_local[i,d]-
                      posiciones[i,d]
                    )
                    +
                    c2*r2*(
                      mejor_global[d]-
                      posiciones[i,d]
                    )
                )

                posiciones[i,d] = (
                    posiciones[i,d]
                    +
                    velocidades[i,d]
                )


            for j in range(max_torres):

                posiciones[i,j*3] = np.clip(
                    posiciones[i,j*3],
                    x_min,
                    x_max
                )

                posiciones[i,j*3+1] = np.clip(
                    posiciones[i,j*3+1],
                    y_min,
                    y_max
                )

                posiciones[i,j*3+2] = np.clip(
                    posiciones[i,j*3+2],
                    0,
                    radio_max
                )


            fit = calcular_fitness(
                posiciones[i]
            )


            if fit > fitness_local[i]:

                fitness_local[i] = fit

                mejor_local[i] = posiciones[i].copy()


                if fit > fitness_global:

                    fitness_global = fit

                    mejor_global = posiciones[i].copy()



    return mejor_global,fitness_global




# -----------------------------------------
# algoritmo genetico
# -----------------------------------------

def optimizacion_genetica(
        poblacion_tam,
        generaciones
):

    dimension = max_torres*3

    poblacion = crear_poblacion(
        poblacion_tam
    )

    mejor = None

    mejor_fitness = -999999



    for g in range(generaciones):

        fitness = np.zeros(
            poblacion_tam
        )


        for i in range(poblacion_tam):

            fitness[i] = calcular_fitness(
                poblacion[i]
            )


        indice = np.argmax(fitness)

        if fitness[indice] > mejor_fitness:

            mejor_fitness = fitness[indice]

            mejor = poblacion[indice].copy()



        nueva = np.zeros_like(
            poblacion
        )

        nueva[0] = mejor.copy()



        for i in range(
            1,
            poblacion_tam
        ):

            p1 = np.random.randint(
                0,
                poblacion_tam
            )

            p2 = np.random.randint(
                0,
                poblacion_tam
            )

            hijo = np.zeros(dimension)


            for j in range(dimension):

                if np.random.rand() < 0.5:
                    hijo[j] = poblacion[p1,j]
                else:
                    hijo[j] = poblacion[p2,j]


            if np.random.rand() < 0.15:

                for j in range(dimension):

                    hijo[j] = hijo[j] + np.random.normal(0,2)


            nueva[i] = hijo


        poblacion = nueva


    return mejor,mejor_fitness




# -----------------------------------------
# grey wolf optimizer
# -----------------------------------------

def optimizacion_gwo(
        lobos,
        iteraciones
):

    dimension = max_torres*3

    posiciones = crear_poblacion(
        lobos
    )

    alfa = posiciones[0].copy()

    mejor_fit = calcular_fitness(alfa)



    for it in range(iteraciones):

        for i in range(lobos):

            fit = calcular_fitness(
                posiciones[i]
            )

            if fit > mejor_fit:

                mejor_fit = fit

                alfa = posiciones[i].copy()



        a = 2 - (2*it/iteraciones)


        for i in range(lobos):

            for d in range(dimension):

                r1=np.random.rand()
                r2=np.random.rand()

                A = 2*a*r1-a
                C = 2*r2

                D = abs(
                    C*alfa[d]
                    -
                    posiciones[i,d]
                )

                posiciones[i,d] = (
                    alfa[d]
                    -
                    A*D
                )


    return alfa,mejor_fit




# -----------------------------------------
# reporte
# -----------------------------------------

def imprimir_reporte(
        solucion,
        fitness,
        tiempo,
        nombre
):

    torres=np.reshape(
        solucion,
        (max_torres,3)
    )

    print("\n")
    print("-------------")
    print(nombre)
    print("-------------")

    print("fitness:",fitness)
    print("tiempo:",tiempo)


    for i in range(max_torres):

        x=torres[i,0]
        y=torres[i,1]
        r=torres[i,2]

        print(
            "torre",
            i+1,
            x,
            y,
            r
        )



# -----------------------------------------
# menu
# -----------------------------------------

def menu():

    while True:

        print("\n1 pso")
        print("2 genetico")
        print("3 gwo")
        print("4 salir")

        op=input("opcion: ")


        if op=="4":
            break


        inicio=time.time()


        if op=="1":

            sol,fit = optimizacion_pso(
                30,
                50
            )

            nombre="pso"


        elif op=="2":

            sol,fit = optimizacion_genetica(
                30,
                50
            )

            nombre="genetico"


        elif op=="3":

            sol,fit = optimizacion_gwo(
                30,
                50
            )

            nombre="gwo"


        else:
            print("opcion incorrecta")
            continue


        tiempo=time.time()-inicio

        imprimir_reporte(
            sol,
            fit,
            tiempo,
            nombre
        )



if __name__=="__main__":
    menu()