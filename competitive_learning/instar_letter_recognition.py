import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, measure, transform
import os
import pyttsx3
import time

# =============================================================================
# Generar patrones individuales
# =============================================================================

ruta = r"Letras.bmp"
carpeta_salida = "letrasmin"

min_pixels = 60   # elimina todo componente con menos de 75 píxeles
tamano_salida = (28, 28)

imagen= io.imread(ruta)
if imagen.shape[2]>3:
    imagen=imagen[:,:,:3]
    imagen= color.rgb2gray(imagen)
    umbral = filters.threshold_otsu(imagen)
    imagen= imagen <umbral

plt.figure()

# 3) Eliminar componentes pequeños
imagen = morphology.remove_small_objects(imagen, min_size=min_pixels)
etiquetas = measure.label(imagen)

plt.imshow(etiquetas)

plt.pause(5)

props = measure.regionprops(etiquetas)

# 5) Calcular cantidad de píxeles por componente
sizes = [p.area for p in props]

# 6) Graficar
plt.figure(figsize=(10,4))
plt.bar(range(1, len(sizes)+1), sizes)
plt.title("Cantidad de píxeles por componente detectado")
plt.xlabel("Índice del componente (sin ordenar)")
plt.ylabel("Pixeles (área real)")
plt.grid(True)
plt.show()

umbral=0.05
# 5) Iterar sobre cada componente
# for i, p in enumerate(props[27:]):
#     minr, minc, maxr, maxc = p.bbox
#     recorte = imagen[minr:maxr, minc:maxc]

#     # Redimensionar a 28x28
#     recorte_resized = transform.resize(recorte.astype(float), tamano_salida, anti_aliasing=True)
#     recorte_resized = recorte_resized > umbral

#     # Mostrar la imagen para etiquetar
#     plt.imshow(recorte_resized, cmap='gray')
#     plt.title(f"Componente {i}/{len(props)}")
#     plt.axis('off')
#     plt.pause(0.1)
#     plt.show()

#     # Pedir nombre
#     letra = input("¿Qué letra es? (ejemplo: a, b, A, B, etc.)\n> ").strip()

#     if letra == "":
#         print("No se ingresó letra, saltando este componente.")
#         continue

#     # Guardar la imagen
#     nombre_archivo = os.path.join(carpeta_salida, f"{letra}.png")
#     io.imsave(nombre_archivo, (recorte_resized * 255).astype(np.uint8))

#     print(f"✅ Guardado: {nombre_archivo}\n")

# print("Proceso completado. Imágenes guardadas en:", carpeta_salida)

# def hardlim(n):
#     out = np.where(n>0,1,0)
#     return out

# def reconocedor(pesos, imagen, polariza):
#     producto = np.dot(pesos, imagen)+polariza
#     out = hardlim(producto)
#     return out

# np.random.seed(42)
# pesos = np.random.rand(784)
# alpha = 0.5
# num_iteraciones = 10000
# # dependiendo de esto es que tanto permitirá que se parezca para que se active la neruona
# polarizacion = -34
# umbral= 0.1
# imagen_a = io.imread('letrasmay/w.png').flatten()
# imagen_a = imagen_a > umbral

# for _ in range(num_iteraciones):
#     b = -np.linalg.norm(pesos) * np.linalg.norm(imagen_a)*0.1
#     n = np.dot(pesos, imagen_a)+b
#     respuesta = hardlim(n)
#     if respuesta == 1:
#         pesos = pesos+alpha*(imagen_a-pesos)

#     if np.linalg.norm(imagen_a - pesos) < 1e-6:
#         break

#     plt.figure(0)
#     plt.imshow(pesos.reshape((28, 28)), cmap='gray')
#     plt.title('pesos entrenados')
#     plt.axis("off")

# =============================================================================
# Training
# =============================================================================

# def hardlim(n):
#     out = np.where(n>0,1,0)
#     return out

# def reconocedor(pesos, imagen, polariza):
#     producto = np.dot(pesos, imagen)+polariza
#     out = hardlim(producto)
#     return out

# # lista de carpetas a recorrer
# carpetas = ["letrasmay", "letrasmin"]

# archivos_totales = []

# np.random.seed(42)
# alpha = 0.5
# num_iteraciones = 10000
# umbral= 0.1


# for carpeta in carpetas:
#     for nombre in os.listdir(carpeta):
#         ruta = os.path.join(carpeta, nombre)
#         if os.path.isfile(ruta):  # solo archivos (no carpetas)
#             archivos_totales.append(ruta)

# print("Archivos encontrados:")
# pesos_tot=[]
# for archivo in archivos_totales:
#     pesos = np.random.rand(784)
#     imagen_a = io.imread(archivo).flatten()
#     imagen_a = imagen_a > umbral
#     for _ in range(num_iteraciones):
#         b = -np.linalg.norm(pesos) * np.linalg.norm(imagen_a)*0.1
#         n = np.dot(pesos, imagen_a)+b
#         respuesta = hardlim(n)
#         if respuesta == 1:
#             pesos = pesos+alpha*(imagen_a-pesos)

#         if np.linalg.norm(imagen_a - pesos) < 1e-6:
#             break

#         plt.figure(0)
#         plt.imshow(pesos.reshape((28, 28)), cmap='gray')
#         plt.title('pesos entrenados')
#         plt.axis("off")
#     pesos=(pesos >= 0.1).astype(float)
#     pesos_tot.append(pesos)

#     # imagen_b = io.imread(archivo).flatten()
#     # imagen_n = imagen_a > umbral
#     # b = -np.linalg.norm(pesos) * np.linalg.norm(imagen_b)*0.9
#     # resultado = reconocedor(pesos, imagen_b, b)
#     # print(archivo,f'resultado{resultado}')

# pesos_tot = np.array(pesos_tot).T
# print(pesos_tot.shape)

# np.save("pesos_letras.npy", pesos_tot)
# print("Pesos guardados")

# imagen_b = io.imread('letrasmay/A.png').flatten()
# imagen_n = imagen_a > umbral
# b = -np.linalg.norm(pesos) * np.linalg.norm(imagen_b)*0.9
# resultado = reconocedor(pesos, imagen_b, b)
# print(f'el resultado de la neurona para la letra "A":{resultado}')

# =============================================================================
# Testing
# =============================================================================

# def hardlim(n):
#     out = np.where(n > 0, 1, 0)
#     return out


# def reconocedor(pesos, imagen, polariza):
#     producto = np.dot(pesos, imagen)+polariza
#     out = hardlim(producto)
#     return out

# letras_com = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# pesos = np.load("pesos_letras.npy")
# print("Pesos cargados")
# print(pesos.shape)

# ruta = r"Letras.bmp"
# carpeta_salida = "letrasmin"

# min_pixels = 75   # elimina todo componente con menos de 75 píxeles
# tamano_salida = (28, 28)

# imagen = io.imread(ruta)
# if imagen.shape[2] > 3:
#     imagen = imagen[:, :, :3]
#     imagen = color.rgb2gray(imagen)
#     umbral = filters.threshold_otsu(imagen)
#     imagen = imagen < umbral

# plt.figure()

# # 3) Eliminar componentes pequeños
# imagen = morphology.remove_small_objects(imagen, min_size=min_pixels)
# etiquetas = measure.label(imagen)

# plt.imshow(etiquetas)

# # plt.pause(5)

# props = measure.regionprops(etiquetas)

# umbral = 0.05
# pes_norm = []
# for i in range(pesos.shape[1]):
#     pes_norm.append(np.linalg.norm(pesos[:, i]))
# pes_norm = np.array(pes_norm)

# np.save("pesnorm.npy", pes_norm)
# print("Pesos normalizados guardados")

# # 5) Iterar sobre cada componente
# for i, p in enumerate(props[:]):
#     minr, minc, maxr, maxc = p.bbox
#     recorte = imagen[minr:maxr, minc:maxc]

#     # Redimensionar a 28x28
#     recorte_resized = transform.resize(recorte.astype(
#         float), tamano_salida, anti_aliasing=True)
#     recorte_resized = recorte_resized > umbral

#     b = -np.dot(pes_norm, np.linalg.norm(recorte_resized.flatten())) * 0.95
#     n = np.dot(pesos.T, recorte_resized.flatten())+b
#     out = hardlim(n)
#     idx=np.argmax(out)
#     rec=letras_com[idx]
#     print(rec)

#     # Mostrar la imagen para etiquetar
#     plt.imshow(recorte_resized, cmap='gray')
#     plt.title(f"Componente {i}/{len(props)}")
#     plt.axis('off')
#     plt.pause(1)
#     plt.show()

#     # resultado = reconocedor(pesos, imagen_b, b)

# =============================================================================
# Poema
# =============================================================================

# def hardlim(n):
#     out = np.where(n > 0, 1, 0)
#     return out


# def reconocedor(pesos, imagen, polariza):
#     producto = np.dot(pesos, imagen)+polariza
#     out = hardlim(producto)
#     return out


# letras_com = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# pesos = np.load("pesos_letras.npy")
# print("Pesos cargados")
# print(pesos.shape)

# pes_norm = np.load("pesos_letras_norm.npy")


# ruta = r"Poema_1.bmp"
# # carpeta_salida = "letrasmin"

# min_pixels = 60   # elimina todo componente con menos de 75 píxeles
# tamano_salida = (28, 28)
# distancia_palabras = 21.2

# imagen = io.imread(ruta)
# if imagen.shape[2] > 3:
#     imagen = imagen[:, :, :3]
#     imagen = color.rgb2gray(imagen)
#     umbral = filters.threshold_otsu(imagen)
#     imagen = imagen < umbral

# # # plt.figure()

# # 3) Eliminar componentes pequeños
# imagen = morphology.remove_small_objects(imagen, min_size=min_pixels)
# etiquetas = measure.label(imagen)

# # # plt.imshow(etiquetas)

# props = measure.regionprops(etiquetas)

# # Extraer bounding boxes de cada componente (carácter)
# boxes = []
# for r in props:
#     minr, minc, maxr, maxc = r.bbox
#     h = maxr - minr
#     w = maxc - minc
#     if h > 0 and w > 0:
#         ycen = 0.5 * (minr + maxr)
#         xcen = 0.5 * (minc + maxc)
#         boxes.append({
#             "minr": minr, "minc": minc, "maxr": maxr, "maxc": maxc,
#             "h": h, "w": w, "ycen": ycen, "xcen": xcen
#         })

# if not boxes:
#     raise ValueError(
#         "No se encontraron componentes. Revisa el umbral o 'min_pixels'.")

# --- Agrupar por renglones (filas) según cercanía vertical ---
# Ordenamos por centro vertical
# boxes.sort(key=lambda b: b["ycen"])
# alturas = np.array([b["h"] for b in boxes])
# # tolerancia vertical para agrupar en la misma fila
# tol = max(10, int(np.median(alturas) * 0.6))

# filas = []  # cada fila será una lista de boxes
# centros_fila = []  # y-centro promedio de la fila para ir actualizando

# for b in boxes:
#     asignada = False
#     for k, ymean in enumerate(centros_fila):
#         if abs(b["ycen"] - ymean) <= tol:
#             filas[k].append(b)
#             # actualizar centro vertical promedio de la fila
#             centros_fila[k] = np.mean([bb["ycen"] for bb in filas[k]])
#             asignada = True
#             break
#     if not asignada:
#         filas.append([b])
#         centros_fila.append(b["ycen"])


# # # --- Para cada fila: ordenar caracteres de izquierda a derecha y medir distancias ---
# distancias_por_fila = []
# todas_las_distancias = []

# for fila in filas:
#     fila.sort(key=lambda b: b["minc"])  # ordenar por x izquierda
#     dists = []
#     for i in range(len(fila) - 1):
#         # distancia entre borde derecho e izquierdo
#         dx = fila[i+1]["minc"] - fila[i]["maxc"]
#         dists.append(max(dx, 0))  # clamp a 0 si hay solapamiento
#     distancias_por_fila.append(dists)
#     todas_las_distancias.extend(dists)

# # --- Visualización: recortes por fila ---
# # Usamos la imagen original en gris para los recortes (si la tienes disponible como 'imagen' en [0,1]/[0,255])
# # Si 'imagen' ya es binaria, también sirve para visualizar las letras.

# n_filas = len(filas)
# if n_filas == 0:
#     raise ValueError(
#         "No se formaron filas. Ajusta 'tol' o el preprocesamiento.")

# # Calcular un padding pequeño para los recortes
# pad = 5

# =========================
# Reconocimiento y armado de palabras por fila
# =========================
# texto_por_fila = []
# texto_completo = []

# for idx_fila, fila in enumerate(filas):
#     # Orden ya hecho: izquierda -> derecha
#     # distancias entre letras de esta fila:
#     dists = distancias_por_fila[idx_fila] if idx_fila < len(
#         distancias_por_fila) else []

#     palabras = []
#     palabra_actual = []

#     for i, box in enumerate(fila):
#         # recorta carácter según bbox del box
#         y0, x0, y1, x1 = box["minr"], box["minc"], box["maxr"], box["maxc"]
#         rec_char = imagen[y0:y1, x0:x1]  # usa 'imagen' binaria o en gris
#         vchar = recorte_resized = transform.resize(rec_char.astype(
#             float), tamano_salida, anti_aliasing=True)
#         recorte_resized = vchar > umbral
#         b = -np.dot(pes_norm, np.linalg.norm(recorte_resized.flatten())) * 0.95
#         n = np.dot(pesos.T, recorte_resized.flatten())+b
#         out = hardlim(n)
#         idx = np.argmax(out)
#         rec = letras_com[idx]

#         # agrega letra actual a la palabra en curso
#         if rec is None:
#             rec = "?"  # por si no reconoce
        
#         palabra_actual.append(rec)

#         # decidir si cerrar palabra: si hay siguiente distancia y es > umbral
#         if i < len(fila) - 1:
#             dist_sig = dists[i] if i < len(dists) else 0
#             if dist_sig > distancia_palabras:
#                 # cerrar palabra actual
#                 palabras.append("".join(palabra_actual))
#                 palabra_actual = []

#     # cierra la última palabra de la fila
#     if palabra_actual:
#         palabras.append("".join(palabra_actual))

#     # reconstruye el texto de la fila
#     texto_linea = " ".join(palabras)
#     texto_por_fila.append(texto_linea)
#     texto_completo.append(texto_linea)

# # Resultado final: todo el texto de golpe
# texto_final = "\n".join(texto_completo)


# print("\n=== TEXTO COMPLETO ===")
# print(texto_final)

# def leer_texto_en_voz_alta(texto_final):
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 175)   # velocidad de lectura (ajusta a gusto)
#     engine.setProperty('volume', 1.0) # volumen máximo

#     # Intentar seleccionar una voz en español
#     for voz in engine.getProperty('voices'):
#         if "es" in voz.id.lower() or "spanish" in voz.name.lower():
#             engine.setProperty('voice', voz.id)
#             break

#     # Leer todo el texto de una sola vez
#     engine.say(texto_final)
#     engine.runAndWait()

# # Uso
# leer_texto_en_voz_alta(texto_final)