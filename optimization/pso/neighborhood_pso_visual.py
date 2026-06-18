import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D   # solo para activar 3D
from matplotlib import cm

# =============================================================================
# Funcion Objetivo
# =============================================================================


def fitness_egg_holder(POS):
    """
    Dominio típico:[-512, 512].
    Gran cantidad de mínimos locales.
    Mínimo global ≈ f(512, 404.2319) ≈ -959.6407.
    """
    X = POS[:, 0]
    Y = POS[:, 1]
    return -(Y + 47) * np.sin(np.sqrt(np.abs(X/2 + (Y + 47)))) \
           - X * np.sin(np.sqrt(np.abs(X - (Y + 47))))


def fitness_himmelblau(POS):
    """
    Dominio típico: [-6, 6].
    Tiene 4 mínimos globales con f(x, y) = 0:
        ( 3.000000,  2.000000)
        (-2.805118,  3.131312)
        (-3.779310, -3.283186)
        ( 3.584428, -1.848126)
    Superficie con varios valles y un máximo local cerca de (-0.27, -0.92).
    """
    X = POS[:, 0]
    Y = POS[:, 1]
    return (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2


def fitness_easom(POS):
    """
    Dominio típico: [-10, 10] (a veces se usa [-100, 100]).
    Casi toda la superficie vale ≈ 0, salvo un pozo muy estrecho.
    Mínimo global: f(π, π) = -1.
    """
    X = POS[:, 0]
    Y = POS[:, 1]
    return -np.cos(X) * np.cos(Y) * np.exp(-(X - np.pi)**2 - (Y - np.pi)**2)


def fitness_styblinski_tang(POS):
    """
    Dominio típico: [-5, 5].
    Multimodal: varios mínimos locales repartidos en el dominio.
    En cada dimensión el mínimo global está en x_i ≈ -2.903534.
    """
    X = POS[:, 0]
    Y = POS[:, 1]
    term_x = X**4 - 16*X**2 + 5*X
    term_y = Y**4 - 16*Y**2 + 5*Y
    return 0.5 * (term_x + term_y)


def fitness_bukin_n6(POS):
    """
    Dominio típico:
        x [-15, -5]
        y [ -3,  3]
    Presenta un valle muy estrecho y curvo: difícil de explorar.
    Mínimo global:
        f(-10, 1) = 0.
    """
    X = POS[:, 0]
    Y = POS[:, 1]
    return 100 * np.sqrt(np.abs(Y - 0.01*X**2)) + 0.01 * np.abs(X + 10)


def fitness_drop_wave(POS):
    """
    Dominio típico: [-5.12, 5.12] (a veces [-10, 10]).
    Superficie radialsimétrica, con muchos mínimos locales concéntricos.
    Mínimo global: f(0, 0) = -1.
    Muy útil para estudiar cómo un algoritmo converge hacia el centro
      entre anillos de mínimos locales.
    """
    X = POS[:, 0]
    Y = POS[:, 1]
    R = np.sqrt(X**2 + Y**2)
    return -(1 + np.cos(12 * R)) / (0.5 * (X**2 + Y**2) + 2)


def fitness_cross_in_tray(POS):
    """
   Dominio típico: [-10, 10].
   Superficie tipo "flor" con múltiples valles profundos y picos.
   Tiene 4 mínimos globales equivalentes con:
       f(x, y) ≈ -2.062612
     en los puntos:
       ( 1.349406685...,  1.349406608...)
       ( 1.349406685..., -1.349406608...)
       (-1.349406685...,  1.349406608...)
       (-1.349406685..., -1.349406608...)
   """
    X = POS[:, 0]
    Y = POS[:, 1]
    exp_term = np.exp(np.abs(100 - np.sqrt(X**2 + Y**2) / np.pi))
    inside = np.sin(X) * np.sin(Y) * exp_term
    return -0.0001 * (np.abs(inside) + 1)


def fitness_egg_crate(POS):
    """
    Función Egg Crate.
    Dominio típico: [-5, 5] o [-2π, 2π].
    Mínimo global:
        f(0, 0) = 0.
    """
    X = POS[:, 0]
    Y = POS[:, 1]
    return X**2 + Y**2 + 25 * (np.sin(X)**2 + np.sin(Y)**2)

# =============================================================================
# Inicialización
# =============================================================================


fitness = fitness_himmelblau

n_pop = 20  # Numero de particulas
dim = 2     # Dimensión del problema
limites = (-6, 6)
n_iter = 100

xmin, xmax = limites
# Matriz de particulas
X = np.random.uniform(xmin, xmax, size=(n_pop, dim))
V = np.zeros((n_pop, dim))

# Al inicio el mejor personal es la misma posición, y
pbest_pos = X.copy()
pbest_val = fitness(X)   # valor de la función en cada partícula

# Obtenemos la primer "mejor global"

# índice de la mejor partícula (minimización)
idx_mejor = np.argmin(pbest_val)
gbest_pos = pbest_pos[idx_mejor].copy()
gbest_val = pbest_val[idx_mejor]

# =============================================================================
# Parametros PSO
# =============================================================================

w_max = 1
w_min = 0.1
c1 = 1
c2 = 1
reset = 20
V_MAX = (xmax - xmin) * 0.2

operador_genetico_de_posicion = True
operador_genetico_de_velocidad = True
reseteo_de_velocidad = True
inercia_variable = True
vecindarios_dinamicos = True

history_best = []  # para guardar el mejor valor por iteración
cont = 0

# =============================
# PREPARAR SUPERFICIE 3D
# =============================
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# OJO: para visualizar, conviene acotar el rango,
# porque [-512, 512] es gigantesco.
plot_min, plot_max = -6, 6  # o por ejemplo -200,200 si quieres ver más detalle

xx = np.linspace(plot_min, plot_max, 100)
yy = np.linspace(plot_min, plot_max, 100)
XX, YY = np.meshgrid(xx, yy)

# armamos POS de la rejilla para pasarla al fitness
POS_grid = np.column_stack([XX.ravel(), YY.ravel()])
ZZ = fitness(POS_grid).reshape(XX.shape)

plt.pause(0.5)


def draw_swarm3d(X, gbest_pos, iteracion):
    ax.clear()

    # superficie
    ax.plot_surface(XX, YY, ZZ, cmap=cm.viridis, alpha=0.7,
                    linewidth=0, antialiased=False)

    # z de las partículas
    Z_part = fitness(X)  # vector (n_pop,)
    ax.scatter(X[:, 0], X[:, 1], Z_part,
               color='blue', s=20, label='Partículas')

    # mejor global
    gbest_z = fitness(gbest_pos.reshape(1, 2))[0]
    ax.scatter(gbest_pos[0], gbest_pos[1], gbest_z,
               color='red', s=80, marker='X', label='Mejor Global')

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_title(f"Iteración {iteracion}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    ax.legend(loc='upper right')

    plt.pause(0.2)


plt.ion()
fig, ax1 = plt.subplots(figsize=(7, 6))

# def draw_swarm2d(X, gbest_pos, iteracion):
#     ax1.clear()
#     ax1.scatter(X[:,0], X[:,1], color='blue', s=20, label='Partículas')
#     ax1.scatter(gbest_pos[0], gbest_pos[1], color='red', s=80, marker='X', label='Mejor Global')
#     ax1.set_xlim(xmin, xmax)
#     ax1.set_ylim(xmin, xmax)
#     ax1.set_title(f"Iteración {iteracion}")
#     ax1.legend()
#     plt.pause(0.05)


for i in range(n_iter+1):
    r1 = np.random.rand(n_pop, dim)
    r2 = np.random.rand(n_pop, dim)

    if inercia_variable:
        # Inercia sera variable y disminuira linealmente
        w = (w_max - w_min)*(n_iter - i)/n_iter + w_min
    else:
        w = 0.5

    # Actualizamos velocidad si no cambia el best
    if reseteo_de_velocidad:
        if cont >= reset:
            r = np.random.rand(n_pop, dim)
            V = V+(2*r-1) * V_MAX
            cont = 0
        else:
            if vecindarios_dinamicos:
                MAXNEIGH = max(1, n_pop // 4)

                indices = np.random.permutation(n_pop)
                neigh = np.array_split(indices, MAXNEIGH)

                best_neigh = np.zeros((MAXNEIGH, dim))
                for ni, group in enumerate(neigh):
                    vals = pbest_val[group]
                    idx_local_best = group[np.argmin(vals)]
                    best_neigh[ni] = pbest_pos[idx_local_best]

                # ==========================================================
                #   ACTUALIZAR VELOCIDAD CON NMPSO
                # ==========================================================
                V_new = np.zeros_like(V)

                for ni, group in enumerate(neigh):
                    pg_kn = best_neigh[ni]
                    for idx in group:
                        r1i = np.random.rand(dim)
                        r2i = np.random.rand(dim)

                        V_new[idx] = (
                            w * V[idx]
                            + c1 * r1i * (pbest_pos[idx] - X[idx])
                            + c2 * r2i * (pg_kn - X[idx])
                        )

                V = V_new.copy()
            else:
                V = w * V + c1 * r1 * (pbest_pos - X) + \
                    c2 * r2 * (gbest_pos - X)
        n_pop, dim = V.shape
    else:
        if vecindarios_dinamicos:
            MAXNEIGH = max(1, n_pop // 4)

            indices = np.random.permutation(n_pop)
            neigh = np.array_split(indices, MAXNEIGH)

            best_neigh = np.zeros((MAXNEIGH, dim))
            for ni, group in enumerate(neigh):
                vals = pbest_val[group]
                idx_local_best = group[np.argmin(vals)]
                best_neigh[ni] = pbest_pos[idx_local_best]

            V_new = np.zeros_like(V)

            for ni, group in enumerate(neigh):
                pg_kn = best_neigh[ni]
                for idx in group:
                    r1i = np.random.rand(dim)
                    r2i = np.random.rand(dim)

                    V_new[idx] = (
                        w * V[idx]
                        + c1 * r1i * (pbest_pos[idx] - X[idx])
                        + c2 * r2i * (pg_kn - X[idx])
                    )

            V = V_new.copy()
        else:
            V = w * V + c1 * r1 * (pbest_pos - X) + c2 * r2 * (gbest_pos - X)

    if operador_genetico_de_velocidad:
        # elegimos una al azar
        k = np.random.randint(0, n_pop)
        # Elegimos dos padres distintos
        idx_par = np.random.choice(n_pop, size=2, replace=False)
        v1 = V[idx_par[0]]   # par1(v_i)
        v2 = V[idx_par[1]]   # par2(v_i)
        s = v1 + v2
        norm_s = np.linalg.norm(s)
        if norm_s == 0:
            # Si la suma es cero (direcciones opuestas), devolvemos v1 tal cual
            v_hijo = v1.copy()
        else:
            direccion = s / norm_s
            magnitud = np.linalg.norm(v1)
            v_hijo = direccion * magnitud

        V[k] = v_hijo

    # Actualizamos la posicion
    X = X+V

    if operador_genetico_de_posicion:
        j = np.random.randint(0, n_pop)          # una partícula al azar
        factor = (n_iter - i) / n_iter           # (MAXITER - iter)/MAXITER
        ruido = np.random.normal(0.0, 1.0, size=dim)  # N(0,1) por componente

        # nueva posición hijo para esa partícula
        X[j] = X[j] + factor * ruido

    # Restringimos los limites (al ser una función cuadrada podemos usar solo dos valores)
    X = np.clip(X, xmin, xmax)

    # Evaluamos la función objetivo
    F = fitness(X)

    # Actualizamos pbest
    mejoro = F < pbest_val  # Aplicamos una mascara booleana
    pbest_pos[mejoro] = X[mejoro]
    pbest_val[mejoro] = F[mejoro]

    # Actualizamos gbest
    idx_mejor = np.argmin(pbest_val)
    if pbest_val[idx_mejor] < gbest_val:
        gbest_val = pbest_val[idx_mejor]
        gbest_pos = pbest_pos[idx_mejor].copy()
        cont = 0
    else:
        cont += 1

    history_best.append(gbest_val)
    # draw_swarm2d(X, gbest_pos, i)
    draw_swarm3d(X, gbest_pos, i)
    # Para ver el progreso
    if i % 10 == 0:
        if inercia_variable:
            print(f"Iteración {i}, mejor valor: {gbest_val:.12f}, con w :f{w}")
        else:
            print(f"Iteración {i}, mejor valor: {gbest_val:.12f}")

idx_mejor = np.argmin(pbest_val)
print(f"Mejor posición encontrada: {pbest_pos[idx_mejor]}")
