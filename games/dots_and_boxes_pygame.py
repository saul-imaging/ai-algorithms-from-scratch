import pygame as pg
from math import hypot

# ---------- Config ----------
DOTS = 5
N = 2*DOTS - 1       # 9 (matriz lógica)
SPACING = 90         # pixeles entre puntos
MARGIN = 60
# Total de puntos
DOT_R = 5
# ancho de linea definitiva
LINE_W = 8
# Ancho de linea de preview
PREVIEW_W = 3
CLICK_TOL = 16       # tolerancia en pixeles para saber que estás seleccionando una linea
SIZE = MARGIN*2 + SPACING*(DOTS-1)
W, H = SIZE, SIZE

# Definimos algunos colores
COL_BG = (248, 250, 252)
COL_DOT = (15, 23, 42)
COL_PREV = (100, 116, 139)
COL_P1 = (37, 99, 235)
COL_P2 = (220, 38, 38)
BOX_FILL_P1 = (191, 219, 254)
BOX_FILL_P2 = (254, 202, 202)

# Iniciamos los modulos de pygame
pg.init()
# desplegamos la pantalla inicial
screen = pg.display.set_mode((W, H))
# Agregamos el nombre de la ventana
pg.display.set_caption("Dots & Boxes 5x5 - Pygame")
# Esto es para optimizar y que no esté usando CPU de más
clock = pg.time.Clock()
# Definimos una fuente para la letra y su tamaño
font = pg.font.SysFont("segoeui", 20)

# ---------- Geometría ----------

# Dibuja los puntos iniciales
# convierte la coordenada del punto (r,c) en la posición en pixeles donde se dibuja el circulo del punto


def dot_xy(r, c):
    x = MARGIN + c*SPACING
    y = MARGIN + r*SPACING
    return x, y


# Matriz 9x9: None=celda no jugable (punto o caja), 0 = arista libre, 1 = J1, 2 = J2
# Regla para filas nones: las unicas columnas jugables son pares
# Regla para filas pares: las unicas columnas jugables son nones
state = [[None]*N for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i % 2 == 0 and j % 2 == 1:   # arista horizontal
            state[i][j] = 0
        elif i % 2 == 1 and j % 2 == 0:  # arista vertical
            state[i][j] = 0
        else:
            state[i][j] = None

# Propietarios de cajas siempre deben ser filas y columnas impares 0 = arista libre, 1 = J1, 2 = J2
boxes = [[None]*N for _ in range(N)]

# Empieza el jugador 1
current_player = 1
# Guardamos los cuadros hechos
scores = {1: 0, 2: 0}
game_over = False
preview = None  # (i,j) de arista bajo el mouse

# ---------- Utilidades ----------

# mapea la celda de la matriz state y la convierte en coordenadas en pixeles


def edge_to_points(i, j):
    if i % 2 == 0 and j % 2 == 1:  # horizontal
        r = i//2
        cL = j//2
        a = (r, cL)
        b = (r, cL+1)
    elif i % 2 == 1 and j % 2 == 0:  # vertical
        c = j//2
        rT = i//2
        a = (rT, c)
        b = (rT+1, c)
    else:
        return None, None
    return dot_xy(*a), dot_xy(*b)


# Es una de las funciones para dibujar el preview de la linea
# Calcula la distancia mas corta entre un punto cualquiera y la linea que está creada por dos puntos
def point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    if dx == 0 and dy == 0:
        return hypot(px-ax, py-ay)  # Distancia euclidiana
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
    qx, qy = ax + t*dx, ay + t*dy
    return hypot(px-qx, py-qy)


# Esta función detecta cual es el borde vacio mas cercano
def nearest_empty_edge(mx, my):
    best = None
    best_d = 1e9
    for i in range(N):
        for j in range(N):
            if state[i][j] == 0:
                a, b = edge_to_points(i, j)
                if not a:
                    continue
                d = point_to_segment_dist(mx, my, *a, *b)
                if d < best_d:
                    best_d, best = d, (i, j)
    return best if best and best_d <= CLICK_TOL else None

# Regresa los bordes de una caja


def box_edges(i, j):
    # i,j impares
    top = (i-1, j)
    bottom = (i+1, j)
    left = (i, j-1)
    right = (i, j+1)
    return (top, bottom, left, right)

# Cuando se aplica una linea se revisan los posibles cuadros que puede completar


def box_completed(i, j):
    for ei, ej in box_edges(i, j):
        if state[ei][ej] in (None, 0):  # no jugable o vacío
            return False
    return True

# Aplica la linea seleccionada a la variable state y revisa si se completó alguna caja


def apply_move(i, j, player):

    state[i][j] = player
    completed = False
    # Checar cajas adyacentes
    adj = []
    if i % 2 == 0 and j % 2 == 1:     # horizontal → cajas arriba/abajo
        if i-1 >= 0:
            adj.append((i-1, j))
        if i+1 < N:
            adj.append((i+1, j))
    elif i % 2 == 1 and j % 2 == 0:   # vertical → cajas izq/der
        if j-1 >= 0:
            adj.append((i, j-1))
        if j+1 < N:
            adj.append((i, j+1))

    for bi, bj in adj:
        if 0 <= bi < N and 0 <= bj < N and bi % 2 == 1 and bj % 2 == 1:
            if boxes[bi][bj] is None and box_completed(bi, bj):
                boxes[bi][bj] = player
                scores[player] += 1
                completed = True
    return completed

# Función que revisa si todas las cajas ya están usadas


def all_boxes_taken():
    total = (DOTS-1)*(DOTS-1)
    return scores[1] + scores[2] == total


# ---------- Dibujo ----------
# recordemos
# coordenada en la matriz estado:
# (par,par)-> punto
# (par, impar) o (impar,par) -> arista
# (impar, impar) -> caja
def draw():
    screen.fill(COL_BG)  # Limpia el fondo
    # rellena cajas ya ganadas
    for i in range(1, N, 2):
        for j in range(1, N, 2):
            owner = boxes[i][j]
            if owner:
                (x1, y1) = dot_xy(i//2, j//2)
                (x2, y2) = dot_xy(i//2+1, j//2+1)
                pad = LINE_W//2 + 3
                rect = pg.Rect(x1+pad, y1+pad, (x2-x1)-2*pad, (y2-y1)-2*pad)
                pg.draw.rect(screen, BOX_FILL_P1 if owner ==
                             1 else BOX_FILL_P2, rect)

    # Dibuja las aristas definitivas
    for i in range(N):
        for j in range(N):
            if state[i][j] in (1, 2):
                a, b = edge_to_points(i, j)
                # Dedice de quien es la linea y el color del jugador
                col = COL_P1 if state[i][j] == 1 else COL_P2
                pg.draw.line(screen, col, a, b, LINE_W)

    # preview solo si la linea está vacia
    if preview and state[preview[0]][preview[1]] == 0:
        a, b = edge_to_points(*preview)
        pg.draw.line(screen, COL_PREV, a, b, PREVIEW_W)

    # puntos lo hacemos desupués para que se vean definidos y por encima de las lineas
    for r in range(DOTS):
        for c in range(DOTS):
            x, y = dot_xy(r, c)
            pg.draw.circle(screen, COL_DOT, (x, y), DOT_R)

    # cambia el texto de la parte superior marcador / turno (simple, encima)
    text = f"P1: {scores[1]}  P2: {scores[2]}   Turno: P{current_player}"
    surf = font.render(text, True, COL_DOT)
    screen.blit(surf, (10, 10))

    pg.display.flip()


# ---------- Loop ----------
running = True
while running:
    # se obtiene la posisicon en pixeles de la ventana la posición del mouse
    mx, my = pg.mouse.get_pos()
    # Nunca debería entrar aqui pero por si acaso
    preview = nearest_empty_edge(mx, my) if not game_over else None

    # Con pg.event.get() obtenemos todos los eventos actuales (cerrar ventana, clics, teclas, etc.)
    for e in pg.event.get():
        if e.type == pg.QUIT:  # Si detecta cerrar ventana cierra el running
            running = False
        # Si se da clic con el boton izquierdo (1) // boton derecho (2) 
        elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1 and not game_over:
            target = nearest_empty_edge(mx, my)
            if target:
                completed = apply_move(*target, current_player)
                if not completed:
                    current_player = 2 if current_player == 1 else 1
                if all_boxes_taken():
                    game_over = True

    draw()
    # Especifica 60 fps para gastar menos memoria
    clock.tick(60)

pg.quit()
