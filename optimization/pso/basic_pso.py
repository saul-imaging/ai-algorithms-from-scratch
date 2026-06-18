import numpy as np

# =============================================================================
# Funcion Objetivo
# =============================================================================


def fitness(POS):
    X = POS[:, 0]
    Y = POS[:, 1]
    return -(Y + 47) * np.sin(np.sqrt(np.abs(X/2 + (Y + 47)))) \
        - X * np.sin(np.sqrt(np.abs(X - (Y + 47))))

# =============================================================================
# Inicialización
# =============================================================================


n_pop = 50  # Numero de particulas
dim = 2     # Dimensión del problema
limites = (-512, 512)
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

w = 1
c1 = 2
c2 = 2

history_best = []  # para guardar el mejor valor por iteración

for i in range(n_iter+1):
    r1 = np.random.rand(n_pop, dim)
    r2 = np.random.rand(n_pop, dim)

    # Actualizamos velocidad
    V = w * V + c1 * r1 * (pbest_pos - X) + c2 * r2 * (gbest_pos - X)

    # Actualizamos la posicion
    X = X+V

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

    history_best.append(gbest_val)

    # Para ver el progreso
    if i % 10 == 0:
        print(f"Iteración {i}, mejor valor: {gbest_val:.12f}")

idx_mejor = np.argmin(pbest_val)
print(f"Mejor posición encontrada: {pbest_pos[idx_mejor]}")