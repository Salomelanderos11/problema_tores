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
# artificial bee colony (abc)
# -----------------------------------------

def optimizacion_abc(
        abejas,
        iteraciones
):

    dimension = max_torres * 3

    limite = 20

    posiciones = crear_poblacion(
        abejas
    )

    fitness = np.zeros(abejas)

    intentos = np.zeros(abejas)


    for i in range(abejas):

        fitness[i] = calcular_fitness(
            posiciones[i]
        )


    mejor_indice = np.argmax(
        fitness
    )

    mejor = posiciones[
        mejor_indice
    ].copy()

    mejor_fit = fitness[
        mejor_indice
    ]


    for it in range(iteraciones):


        # abejas empleadas
        for i in range(abejas):

            vecino = np.random.randint(
                0,
                abejas
            )

            while vecino == i:
                vecino = np.random.randint(
                    0,
                    abejas
                )


            nueva = posiciones[i].copy()

            torre = np.random.randint(
                0,
                max_torres
            )

            inicio = torre*3


            for k in range(3):

                phi = np.random.uniform(
                    -1,
                    1
                )

                nueva[inicio+k] = (
                    nueva[inicio+k]
                    +
                    phi*
                    (
                      posiciones[i,inicio+k]
                      -
                      posiciones[vecino,inicio+k]
                    )
                )


            fit_nuevo = calcular_fitness(
                nueva
            )


            if fit_nuevo > fitness[i]:

                posiciones[i] = nueva

                fitness[i] = fit_nuevo

                intentos[i]=0

            else:

                intentos[i]=intentos[i]+1



        # abejas exploradoras
        for i in range(abejas):

            if intentos[i] > limite:

                posiciones[i] = crear_poblacion(
                    1
                )[0]

                fitness[i] = calcular_fitness(
                    posiciones[i]
                )

                intentos[i]=0



        indice=np.argmax(fitness)

        if fitness[indice] > mejor_fit:

            mejor_fit=fitness[indice]

            mejor=posiciones[indice].copy()



    return mejor,mejor_fit


# -----------------------------------------
# artificial immune system
# -----------------------------------------

def optimizacion_ais(
        poblacion,
        iteraciones
):

    base = poblacion//3

    if base < 10:
        base = 10


    dimension=max_torres*3

    clones=3


    anticuerpos = crear_poblacion(
        base
    )


    mejor = anticuerpos[0].copy()

    mejor_fit = calcular_fitness(
        mejor
    )



    for it in range(iteraciones):

        afinidad=np.zeros(base)


        for i in range(base):

            afinidad[i]=calcular_fitness(
                anticuerpos[i]
            )


        indice=np.argmax(afinidad)

        if afinidad[indice] > mejor_fit:

            mejor_fit=afinidad[indice]

            mejor=anticuerpos[indice].copy()



        nuevos=[]


        for i in range(base):

            tasa=(i+1)/base


            for c in range(clones):

                clon=anticuerpos[i].copy()


                if np.random.rand()<0.8:

                    for j in range(dimension):

                        cambio=np.random.normal(
                            0,
                            3*tasa
                        )

                        clon[j]=clon[j]+cambio


                nuevos.append(clon)



        nuevos=np.array(nuevos)

        total=len(nuevos)

        fitness_nuevos=np.zeros(total)


        for i in range(total):

            fitness_nuevos[i]=calcular_fitness(
                nuevos[i]
            )


        indices=np.argsort(
            fitness_nuevos
        )

        indices=indices[::-1]


        sobrevivientes=np.zeros(
            (base,dimension)
        )


        for i in range(base):

            sobrevivientes[i]=nuevos[
                indices[i]
            ]


        anticuerpos=survivivientes = sobrevivientes



        reemplazos=int(base*0.2)

        for i in range(reemplazos):

            indice=base-1-i

            anticuerpos[indice]=crear_poblacion(
                1
            )[0]



    return mejor,mejor_fit

# -----------------------------------------
# evaluar desempeño estadistico
# -----------------------------------------

def evaluar_algoritmo(
        funcion_algoritmo,
        repeticiones,
        poblacion,
        iteraciones
):

    resultados = []


    print("\nEjecutando pruebas...")
    print("-"*50)


    for i in range(repeticiones):

        solucion,fitness = funcion_algoritmo(
            poblacion,
            iteraciones
        )

        resultados.append(
            fitness
        )

        print(
            "corrida",
            i+1,
            "fitness =",
            round(fitness,4)
        )


    resultados = np.array(
        resultados
    )


    promedio = np.mean(
        resultados
    )

    desviacion = np.std(
        resultados
    )

    mejor = np.max(
        resultados
    )

    peor = np.min(
        resultados
    )


    print("\n")
    print("="*65)
    print("REPORTE ESTADISTICO")
    print("="*65)

    print(
      "numero de corridas: ",
      repeticiones
    )

    print(
      "mejor fitness:     ",
      round(mejor,4)
    )

    print(
      "peor fitness:      ",
      round(peor,4)
    )

    print(
      "promedio fitness:  ",
      round(promedio,4)
    )

    print(
      "desviacion estandar:",
      round(desviacion,4)
    )


    if desviacion < 50:
        print("estabilidad: alta")

    elif desviacion < 200:
        print("estabilidad: media")

    else:
        print("estabilidad: baja")


    print("="*65)

# -----------------------------------------
# reporte
# -----------------------------------------

def imprimir_reporte(
        solucion,
        fitness,
        tiempo,
        nombre
):

    torres = np.reshape(
        solucion,
        (max_torres,3)
    )

    print("\n")
    print("="*72)
    print("         REPORTE FINAL DE OPTIMIZACION".center(72))
    print("="*72)

    print(f" algoritmo ejecutado : {nombre.upper()}")
    print(f" fitness obtenido    : {fitness:10.4f}")
    print(f" tiempo ejecucion    : {tiempo:10.2f} segundos")

    print("-"*72)

    print(
      "{:<8}{:<14}{:<14}{:<14}{:<15}".format(
          "torre",
          "coord x",
          "coord y",
          "radio",
          "estado"
      )
    )

    print("-"*72)


    activas = 0


    for i in range(max_torres):

        x = torres[i][0]
        y = torres[i][1]
        r = torres[i][2]

        if r >= 0.5:
            estado="activa"
            activas +=1
        else:
            estado="apagada"


        print(
            "{:<8}{:<14.2f}{:<14.2f}{:<14.2f}{:<15}".format(
                i+1,
                x,
                y,
                r,
                estado
            )
        )


    print("-"*72)

    print(
      " torres activas:",
      activas,
      "de",
      max_torres
    )

    print("="*72)



# -----------------------------------------
# menu
# -----------------------------------------

def menu():

    while True:

        print("\n")
        print("="*60)
        print("   SISTEMA BIOINSPIRADO PARA UBICACION DE TORRES")
        print("="*60)

        print("  opcion   algoritmo")
        print("  ------   ------------------------------")
        print("    1      particle swarm optimization")
        print("    2      algoritmo genetico")
        print("    3      grey wolf optimizer")
        print("    4      artificial bee colony")
        print("    5      artificial immune system")
        print("    6      pruebas estadisticas")
        print("    7      Salir")
        print("-"*60)

        op = input(" selecciona opcion: ")


        if op=="7":
            print("\ncerrando programa...")
            break


        inicio=time.time()



        if op=="1":

            print("\n ejecutando pso...\n")

            sol,fit = optimizacion_pso(
                30,
                50
            )

            nombre="pso"



        elif op=="2":

            print("\n ejecutando genetico...\n")

            sol,fit = optimizacion_genetica(
                30,
                50
            )

            nombre="genetico"



        elif op=="3":

            print("\n ejecutando gwo...\n")

            sol,fit = optimizacion_gwo(
                30,
                50
            )

            nombre="gwo"



        elif op=="4":

            print("\n ejecutando abc...\n")

            sol,fit = optimizacion_abc(
                30,
                50
            )

            nombre="abc"



        elif op=="5":

            print("\n ejecutando ais...\n")

            sol,fit = optimizacion_ais(
                30,
                50
            )

            nombre="ais"

        elif op=="6":

            print("\n")
            print("1 pso")
            print("2 genetico")
            print("3 gwo")
            print("4 abc")
            print("5 ais")
            print("6 pruebas estadisticas")
            print("7 salir")

            prueba=input(
            "algoritmo para evaluar: "
            )


            if prueba=="1":

                evaluar_algoritmo(
                    optimizacion_pso,
                    30,
                    30,
                    50
                )


            elif prueba=="2":

                evaluar_algoritmo(
                    optimizacion_genetica,
                    30,
                    30,
                    50
                )


            elif prueba=="3":

                evaluar_algoritmo(
                    optimizacion_gwo,
                    30,
                    30,
                    50
                )


            elif prueba=="4":

                evaluar_algoritmo(
                    optimizacion_abc,
                    30,
                    30,
                    50
                )


            elif prueba=="5":

                evaluar_algoritmo(
                    optimizacion_ais,
                    30,
                    30,
                    50
                )


            input(
            "\npresiona enter para volver..."
            )

            continue

        else:

            print("\n opcion no valida\n")
            continue



        tiempo = time.time()-inicio


        imprimir_reporte(
            sol,
            fit,
            tiempo,
            nombre
        )


        input(
           "\npresiona enter para volver al menu..."
        )


if __name__=="__main__":
    menu()