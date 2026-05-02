
# -----------------------------------------
#           INTEGRANTES
#
#   *CAMBEROS BUSTAMANTE JOHANA
#   *LANDEROS SALAZAR SALOME
#   *ROSALES LOPEZ FERNANDO
#
# -----------------------------------------




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

grid_x, grid_y = np.meshgrid(linea_x, linea_y)

area_punto = ((x_max-x_min)/resolucion)*((y_max-y_min)/resolucion)


# -----------------------------------------
# fitness
# -----------------------------------------

def calcular_fitness(individuo):

    torres = np.reshape(individuo,(max_torres,3))

    costo_total = 0
    penalizacion = 0

    mapa = np.zeros((resolucion,resolucion))


    for torre in torres:

        x,y,r = torre

        if r < 0.5:
            continue

        if r > radio_max:
            r = radio_max

        if x < x_min or x > x_max:
            penalizacion += penalizacion_fuera
            continue

        if y < y_min or y > y_max:
            penalizacion += penalizacion_fuera
            continue

        costo_total += np.pi*(r*r)

        dist = (grid_x-x)**2 + (grid_y-y)**2
        mapa += (dist <= r*r)


    area_cubierta = np.sum(mapa>=1)*area_punto
    solapamiento = np.sum(np.maximum(0,mapa-1))*area_punto

    if area_cubierta > 0:
        porcentaje = (solapamiento/area_cubierta)*100
        if porcentaje > max_solapamiento:
            penalizacion += penalizacion_fuera


    penal_factor = 1.5

    fitness = (area_cubierta -penal_factor*( peso_costo*costo_total + peso_solapamiento*solapamiento ) - penalizacion)

    return fitness


# -----------------------------------------
# poblacion
# -----------------------------------------

def crear_poblacion(n):

    dim = max_torres*3

    pob = np.random.uniform(0,100,(n,dim))

    for i in range(n):
        for j in range(max_torres):
            pob[i,j*3+2] = np.random.uniform(0,radio_max)

    return pob


# -----------------------------------------
# PSO
# -----------------------------------------

def optimizacion_pso(n,iters):

    dim = max_torres*3

    w,c1,c2 = 0.7,1.5,1.5

    pos = crear_poblacion(n)
    vel = np.zeros((n,dim))

    pbest = pos.copy()
    pfit = np.array([calcular_fitness(p) for p in pos])

    gbest = pbest[np.argmax(pfit)].copy()
    gfit = np.max(pfit)

    for _ in range(iters):

        for i in range(n):

            for d in range(dim):

                r1,r2 = np.random.rand(),np.random.rand()

                vel[i,d] = (
                    w*vel[i,d]
                    + c1*r1*(pbest[i,d]-pos[i,d])
                    + c2*r2*(gbest[d]-pos[i,d])
                )

                pos[i,d] += vel[i,d]


            for j in range(max_torres):
                pos[i,j*3]   = np.clip(pos[i,j*3],x_min,x_max)
                pos[i,j*3+1] = np.clip(pos[i,j*3+1],y_min,y_max)
                pos[i,j*3+2] = np.clip(pos[i,j*3+2],0,radio_max)


            fit = calcular_fitness(pos[i])

            if fit > pfit[i]:
                pfit[i]=fit
                pbest[i]=pos[i].copy()

                if fit > gfit:
                    gfit=fit
                    gbest=pos[i].copy()

    return gbest,gfit


# -----------------------------------------
# GA
# -----------------------------------------

def optimizacion_genetica(n,iters):

    dim = max_torres*3

    pob = crear_poblacion(n)

    mejor = None
    mejor_fit = -1e9

    for _ in range(iters):

        fit = np.array([calcular_fitness(p) for p in pob])

        idx = np.argmax(fit)

        if fit[idx] > mejor_fit:
            mejor_fit = fit[idx]
            mejor = pob[idx].copy()

        nueva = np.zeros_like(pob)
        nueva[0] = mejor.copy()

        for i in range(1,n):

            p1,p2 = np.random.randint(0,n),np.random.randint(0,n)

            hijo = np.where(np.random.rand(dim)>0.5,pob[p1],pob[p2])

            if np.random.rand()<0.15:
                hijo += np.random.normal(0,2,dim)

            nueva[i]=hijo

        pob=nueva

    return mejor,mejor_fit


# -----------------------------------------
# GWO
# -----------------------------------------

def optimizacion_gwo(n,iters):

    dim = max_torres*3

    pos = crear_poblacion(n)

    alfa = pos[0].copy()
    mejor = calcular_fitness(alfa)

    for it in range(iters):

        for i in range(n):
            f=calcular_fitness(pos[i])
            if f>mejor:
                mejor=f
                alfa=pos[i].copy()

        a = 2-(2*it/iters)

        for i in range(n):
            for d in range(dim):

                r1,r2 = np.random.rand(),np.random.rand()

                A=2*a*r1-a
                C=2*r2

                D=abs(C*alfa[d]-pos[i,d])

                pos[i,d]=alfa[d]-A*D

    return alfa,mejor


# -----------------------------------------
# ABC
# -----------------------------------------

def optimizacion_abc(n,iters):

    dim=max_torres*3

    pos=crear_poblacion(n)
    fit=np.array([calcular_fitness(p) for p in pos])
    intentos=np.zeros(n)

    mejor=pos[np.argmax(fit)].copy()
    mejor_fit=np.max(fit)

    for _ in range(iters):

        for i in range(n):

            k=np.random.randint(0,n)
            while k==i:
                k=np.random.randint(0,n)

            v=pos[i].copy()

            t=np.random.randint(max_torres)

            for j in range(3):
                phi=np.random.uniform(-1,1)
                idx=t*3+j
                v[idx]+=phi*(pos[i,idx]-pos[k,idx])

            f=calcular_fitness(v)

            if f>fit[i]:
                pos[i]=v
                fit[i]=f
                intentos[i]=0
            else:
                intentos[i]+=1

        idx=np.argmax(fit)

        if fit[idx]>mejor_fit:
            mejor_fit=fit[idx]
            mejor=pos[idx].copy()

    return mejor,mejor_fit


# -----------------------------------------
# AIS 
# -----------------------------------------
def optimizacion_ais(n,iters):

    base=max(10,n//3)
    dim=max_torres*3

    anticuerpos=crear_poblacion(base)

    mejor=anticuerpos[0].copy()
    mejor_fit=calcular_fitness(mejor)

    for _ in range(iters):

        fit=np.array(
        [calcular_fitness(a)
         for a in anticuerpos]
        )

        idx=np.argsort(fit)[::-1]

        anticuerpos=anticuerpos[idx]
        fit=fit[idx]

        if fit[0]>mejor_fit:
            mejor_fit=fit[0]
            mejor=anticuerpos[0].copy()

        nuevos=[]

        for i in range(base):

            tasa=(i+1)/base

            for _ in range(3):

                clon=anticuerpos[i].copy()

                if np.random.rand()<0.8:
                    clon+=np.random.normal(
                      0,
                      3*tasa,
                      dim
                    )

                for j in range(max_torres):
                    clon[j*3]=np.clip(
                       clon[j*3],x_min,x_max
                    )
                    clon[j*3+1]=np.clip(
                       clon[j*3+1],y_min,y_max
                    )
                    clon[j*3+2]=np.clip(
                       clon[j*3+2],0,radio_max
                    )

                nuevos.append(clon)

        nuevos=np.array(nuevos)

        fit_n=np.array(
        [calcular_fitness(c)
         for c in nuevos]
        )

        mejores=np.argsort(
          fit_n
        )[::-1][:base]

        anticuerpos=nuevos[mejores]

    return mejor,mejor_fit


# -----------------------------------------
# evaluacion
# -----------------------------------------

def evaluar_algoritmo(func,nombre,reps,n,iters):

    resultados=[]

    for i in range(reps):

        _,f=func(n,iters)

        resultados.append(f)

        print("corrida",i+1,f)

    resultados=np.array(resultados)

    print("\npromedio:",np.mean(resultados))
    print("desv:",np.std(resultados,ddof=1))

    return resultados


# -----------------------------------------
# grafica
# -----------------------------------------

def graficar(resultados,nombre):

    media=np.mean(resultados)
    desv=np.std(resultados,ddof=1)

    plt.figure(figsize=(10,6))

    conteos,bins,_=plt.hist(resultados,bins=12,edgecolor="black")

    x=np.linspace(min(resultados),max(resultados),300)
    ancho=bins[1]-bins[0]

    y=(1/(desv*np.sqrt(2*np.pi)))*np.exp(-(x-media)**2/(2*desv**2))
    y=y*len(resultados)*ancho

    plt.plot(x,y)

    plt.title("Distribucion "+nombre)
    plt.xlabel("Fitness")
    plt.ylabel("Frecuencia")

    plt.show()


# -----------------------------------------
# reporte simple
# -----------------------------------------

def imprimir_reporte(sol,fit,t,nombre):

    print("\n--- RESULTADO ---")
    print("algoritmo:",nombre)
    print("fitness:",fit)
    print("tiempo:",t)


# -----------------------------------------
# menu
# -----------------------------------------

def main():

    while True:

        print("\n1 PSO\n2 GA\n3 GWO\n4 ABC\n5 AIS\n6 salir")

        op=input("opcion: ")

        if op=="6":
            break

        alg={
            "1":(optimizacion_pso,"PSO"),
            "2":(optimizacion_genetica,"GA"),
            "3":(optimizacion_gwo,"GWO"),
            "4":(optimizacion_abc,"ABC"),
            "5":(optimizacion_ais,"AIS")
        }

        if op not in alg:
            continue

        f,nombre=alg[op]

        modo=input("1 corrida / 2 estadistico: ")

        n=int(input("poblacion: "))
        iters=int(input("iteraciones: "))

        if modo=="1":

            t0=time.time()
            sol,fit=f(n,iters)
            t=time.time()-t0

            imprimir_reporte(sol,fit,t,nombre)

        else:

            reps=int(input("corridas: "))

            res=evaluar_algoritmo(f,nombre,reps,n,iters)

            if input("graficar? s/n: ")=="s":
                graficar(res,nombre)


if __name__=="__main__":
    main()