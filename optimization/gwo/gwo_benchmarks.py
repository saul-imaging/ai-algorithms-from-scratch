import numpy as np

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

def sphere(x):
    return np.sum(x**2, axis=1)

ite_max = 50
total_pop = 30
dim = 2

lb = -10
ub = 10


X = np.random.uniform(lb, ub, size=(total_pop, dim))

# ==========================
# Fitness inicial
# ==========================
fitness_func=fitness_easom
fitness = fitness_func(X)

idx_orden = np.argsort(fitness)
idx_alpha = idx_orden[0]
idx_beta  = idx_orden[1]
idx_delta = idx_orden[2]

X_alpha = X[idx_alpha].copy()
X_beta  = X[idx_beta].copy()
X_delta = X[idx_delta].copy()

f_alpha = fitness[idx_alpha]

# ==========================
# Bucle principal
# ==========================
for i in range(ite_max):
    
    # Parámetro a
    a = 2 - 2 * i / ite_max  

    # r1 y r2
    r1_alpha = np.random.rand(total_pop, dim)
    r2_alpha = np.random.rand(total_pop, dim)
    r1_beta  = np.random.rand(total_pop, dim)
    r2_beta  = np.random.rand(total_pop, dim)
    r1_delta = np.random.rand(total_pop, dim)
    r2_delta = np.random.rand(total_pop, dim)

    A1 = 2 * a * r1_alpha - a
    C1 = 2 * r2_alpha

    A2 = 2 * a * r1_beta - a
    C2 = 2 * r2_beta

    A3 = 2 * a * r1_delta - a
    C3 = 2 * r2_delta

    X_alpha_mat = np.tile(X_alpha, (total_pop, 1))
    X_beta_mat  = np.tile(X_beta,  (total_pop, 1))
    X_delta_mat = np.tile(X_delta, (total_pop, 1))

    # USAR X, NO x
    D_alpha = np.abs(C1 * X_alpha_mat - X)
    D_beta  = np.abs(C2 * X_beta_mat  - X)
    D_delta = np.abs(C3 * X_delta_mat - X)

    X1 = X_alpha_mat - A1 * D_alpha
    X2 = X_beta_mat  - A2 * D_beta
    X3 = X_delta_mat - A3 * D_delta

    X = (X1 + X2 + X3) / 3.0

    # Limitar
    X = np.clip(X, lb, ub)

    # ACTUALIZAR FITNESS
    fitness = fitness_func(X)

    # Actualizar jerarquía
    idx_orden = np.argsort(fitness)
    idx_alpha = idx_orden[0]
    idx_beta  = idx_orden[1]
    idx_delta = idx_orden[2]

    X_alpha = X[idx_alpha].copy()
    X_beta  = X[idx_beta].copy()
    X_delta = X[idx_delta].copy()

    f_alpha = fitness[idx_alpha]

    print(f"Iter {i} | Mejor valor: {f_alpha:.6f} | Mejor pos: {X_alpha}")
