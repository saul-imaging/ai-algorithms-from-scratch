import numpy as np

# Número total de estaciones
N = 28

# Crear matriz de distancias inicializada con infinito
D = np.full((N, N), np.inf)

# Distancia 0 en la diagonal
np.fill_diagonal(D, 0.0)

def pedir_estacion(texto):
    """Pide un número de estación válido entre 1 y 28."""
    while True:
        try:
            est = int(input(texto))
            if 1 <= est <= N:
                return est
            else:
                print("❌ Debe ser un número entre 1 y 28.")
        except ValueError:
            print("❌ Entrada inválida. Intenta de nuevo.")

def pedir_distancia():
    """Pide una distancia positiva."""
    while True:
        try:
            d = float(input("Distancia (en metros): "))
            if d >= 0:
                return d
            else:
                print("❌ La distancia debe ser positiva.")
        except ValueError:
            print("❌ Solo se aceptan números.")

print("=== Construcción de Matriz de Distancias ===")

while True:
    print("\n--------------------------------")
    
    est1 = pedir_estacion("Primera estación (1-28): ")
    
    # Segunda estación no puede ser la misma
    while True:
        est2 = pedir_estacion("Segunda estación (1-28, distinta): ")
        if est2 != est1:
            break
        print("❌ No puede ser la misma estación.")

    dist = pedir_distancia()

    # Convertir a índices Python
    i = est1 - 1
    j = est2 - 1

    # Asignar distancia en ambas direcciones
    D[i, j] = dist
    D[j, i] = dist

    print(f"✔ Distancia registrada entre {est1} y {est2}: {dist} m")

    # Preguntar si desea continuar
    seguir = input("\n¿Deseas agregar otra distancia? (s/n): ").strip().lower()
    if seguir != "s":
        break

print("\n=== Matriz final de distancias ===")
print(D)

# Guardar automáticamente
np.save("matriz_distancias.npy", D)
print("\n✔ Matriz guardada como 'matriz_distancias.npy'")
