import numpy as np
import json
import matplotlib.pyplot as plt
import time


def estacion_mas_cercana(punto, coords_lons, coords_lats):
    """Regresa el índice 0-based de la estación más cercana a un punto (lon, lat)."""
    estaciones = np.column_stack([coords_lons, coords_lats])  # (lon, lat)
    dist = np.linalg.norm(estaciones - punto, axis=1)
    return int(np.argmin(dist))   # índice 0-based


class AntColony:
    def __init__(self,
                 coords,
                 conexiones,
                 nombres,
                 d,
                 n_hor=1000,
                 n_iter=200,
                 alpha=1.0,
                 beta=2.0,
                 t_evap=0.1,
                 cant_fer=1.0):
        # Datos de red
        self.coords = coords              # (n, 2)
        self.conexiones = conexiones      # lista de vecinos 0-based
        self.nombres = nombres
        self.d = d                        # matriz de distancias (n x n)

        # Parámetros ACO
        self.n_hor = n_hor
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.t_evap = t_evap
        self.cant_fer = cant_fer

        # Estructuras internas
        self.n = d.shape[0]

        # Matriz de feromonas
        self.feromonas = np.full((self.n, self.n), 0.5, dtype=float)

        # Matriz heurística (1/distancia donde hay conexión válida)
        self.heuristica = np.zeros((self.n, self.n), dtype=float)
        mask = np.isfinite(self.d) & (self.d > 0)
        self.heuristica[mask] = 1.0 / self.d[mask]

        # Mejor solución global
        self.best_cost = np.inf
        self.best_solution = None

        # Para graficar
        self.lats = self.coords[:, 0]
        self.lons = self.coords[:, 1]

    # -------------------------------------------------------------------------
    # Parte de ACO
    # -------------------------------------------------------------------------
    def run(self, origen, destino, conv=20):
        """
        Ejecuta el ACO desde 'origen' a 'destino' (índices 0-based).
        conv: máximo de hormigas seguidas sin mejora de la mejor ruta.
        """
        self.best_cost = np.inf
        self.best_solution = None

        t0 = time.perf_counter()

        # Por si no encuentra ninguna solución volvemos a intentar rápidamente
        while self.best_solution is None:
            for i in range(self.n_iter):
                rutas = []
                costos = []
                actualizacion = 0
                cont = 0

                for horm in range(self.n_hor):
                    actual = origen
                    ruta = [origen]
                    costo = 0.0

                    # Caminamos hasta llegar al destino
                    while actual != destino:
                        candidatos = self.conexiones[actual]
                        valores = []

                        # visibilidad = (fer)^alpha * (heu)^beta
                        for j in candidatos:
                            fer = self.feromonas[actual, j]
                            heu = self.heuristica[actual, j]
                            visibilidad = (fer**self.alpha) * (heu**self.beta)
                            valores.append(visibilidad)

                        suma = sum(valores)
                        # Caso raro: todas las visibilidades salen 0
                        if suma <= 0 or not np.isfinite(suma):
                            costo = np.inf
                            break

                        probs = [v/suma for v in valores]

                        # Elegimos vecino según distribución de probabilidad
                        siguiente = np.random.choice(candidatos, p=probs)

                        # Evitar repetir nodos
                        if siguiente not in ruta:
                            ruta.append(siguiente)
                            costo += self.d[actual, siguiente]
                            actual = siguiente
                        else:
                            costo = np.inf

                        # Podar si ya es peor que la mejor conocida
                        if costo >= self.best_cost:
                            break

                    # Observador de actualización de mejor ruta
                    if costo < self.best_cost:
                        self.best_cost = costo
                        self.best_solution = ruta
                        actualizacion = 1

                    # Conteo para convergencia
                    if actualizacion == 1:
                        cont = 0
                    else:
                        cont += 1

                    if cont > conv:
                        break

                    rutas.append(ruta)
                    costos.append(costo)

                if cont > conv:
                    break

                # Evaporación
                self.feromonas *= (1.0 - self.t_evap)

                # Depósito de feromonas solo para las rutas efectivamente generadas
                for h in range(len(costos)):
                    Lh = costos[h]
                    if Lh == 0 or Lh == np.inf:
                        continue

                    deposito = self.cant_fer / Lh
                    ruta_h = rutas[h]

                    for a, b in zip(ruta_h[:-1], ruta_h[1:]):
                        self.feromonas[a, b] += deposito
                        self.feromonas[b, a] += deposito  # bidireccional

        t1 = time.perf_counter()
        elapsed = t1 - t0
        return self.best_solution, self.best_cost, elapsed

    # -------------------------------------------------------------------------
    # Parte de visualización
    # -------------------------------------------------------------------------
    def seleccionar_origen_destino(self):
        """Muestra el grafo y permite seleccionar dos estaciones con el mouse."""
        plt.figure()
        plt.title("Selecciona dos estaciones con el mouse (click)")

        # dibujar aristas
        for i, vecinos in enumerate(self.conexiones):
            x1, y1 = self.lons[i], self.lats[i]
            for j in vecinos:
                if j > i:  # evitar duplicar aristas
                    x2, y2 = self.lons[j], self.lats[j]
                    plt.plot([x1, x2], [y1, y2])

        # nodos
        plt.scatter(self.lons, self.lats)

        # etiquetas
        for idx, (x, y) in enumerate(zip(self.lons, self.lats), start=1):
            plt.text(x, y, f"{idx}\n{self.nombres[idx-1]}", fontsize=6)

        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.axis("equal")
        plt.tight_layout()
        pts = plt.ginput(2)   # clicks
        plt.close()

        p_origen = np.array(pts[0])
        p_destino = np.array(pts[1])

        origen = estacion_mas_cercana(p_origen, self.lons, self.lats)
        destino = estacion_mas_cercana(p_destino, self.lons, self.lats)

        return origen, destino

    def plot_best_route(self, best_solution, best_cost, elapsed):
        """Dibuja el grafo y resalta la mejor ruta encontrada."""
        best_edges = set()
        for a, b in zip(best_solution, best_solution[1:]):
            e = tuple(sorted((int(a), int(b))))
            best_edges.add(e)

        plt.figure()
        plt.title(f"Grafo de mejor ruta\nCosto: {best_cost:.3f}  Tiempo: {elapsed:.4f}s")

        # dibujar aristas
        for i, vecinos in enumerate(self.conexiones):
            x1, y1 = self.lons[i], self.lats[i]
            for j in vecinos:
                if j > i:
                    x2, y2 = self.lons[j], self.lats[j]
                    e = tuple(sorted((i, j)))
                    if e in best_edges:
                        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2.5, zorder=3)
                    else:
                        plt.plot([x1, x2], [y1, y2], color='gray',
                                 linewidth=0.8, zorder=1)

        # nodos
        plt.scatter(self.lons, self.lats, color='black', s=15, zorder=4)

        # etiquetas
        for idx, (x, y) in enumerate(zip(self.lons, self.lats), start=1):
            plt.text(x, y, f"{idx}\n{self.nombres[idx-1]}",
                     fontsize=6, zorder=5)

        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Cargar datos
    coords = np.load("coordenadas.npy")

    with open("conexiones.json") as f:
        conexiones = json.load(f)
    conexiones = [[c - 1 for c in fila] for fila in conexiones]  # 0-based

    with open("nombres.json") as f:
        nombres = np.array(json.load(f))

    d = np.load("matriz_distancias.npy")

    # Crear ACO
    aco = AntColony(coords, conexiones, nombres, d,
                         n_hor=1000, n_iter=200,
                         alpha=1, beta=2, t_evap=0.1, cant_fer=1)

    # Elegir origen y destino con el mouse
    origen, destino = aco.seleccionar_origen_destino()

    # Ejecutar ACO
    best_solution, best_cost, elapsed = aco.run(origen, destino, conv=50)

    print("Mejor ruta (índices 0-based):", best_solution)
    print("Mejor costo:", best_cost)
    print(f"Tiempo ACO: {elapsed:.4f} segundos")

    # Graficar mejor ruta
    aco.plot_best_route(best_solution, best_cost, elapsed)
