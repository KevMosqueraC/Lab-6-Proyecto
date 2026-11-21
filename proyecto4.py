# -*- coding: utf-8 -*-
# Simulador 3D de un brazo robotico de 5 articulaciones con garra
# Proyecto de Programacion 3
# Librerias: Tkinter (ventana y deslizadores), PyOpenGL (grafico 3D)
# y NumPy (matrices).
#
# Como funciona (resumen para la sustentacion):
# 1. Las piezas se exportaron de SolidWorks como STL desde el ensamble,
#    por eso ya vienen en su posicion correcta (en mm) y no hay que moverlas.
# 2. Cada articulacion se define con un PIVOTE (un punto) y un EJE (una
#    direccion). Los obtuvimos de la geometria de los STL (centro de los
#    pasadores y agujeros).
# 3. Al mover un deslizador se rota alrededor de ese pivote y ese eje la
#    pieza que se mueve y todas las que van despues en la cadena.
#
# Articulaciones:
#   J1 base (eje Y, giro vertical)
#   J2 hombro (eje Z)
#   J3 codo (eje Z)
#   J4 giro del antebrazo (eje a lo largo del tubo verde)
#   J5 muneca (pasador entre la horquilla rosa y la pieza roja)
#   Garra: Part7 y Part7a, un solo deslizador para las dos
#
# Carpetas necesarias junto a este archivo:
#   modelos/     los 8 STL (Part1 ... Part7 y Part7a)
#   ensamblaje/  se crea sola y guarda config_v4.json
import os
import re
import json
import ctypes
import struct
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from pyopengltk import OpenGLFrame
from OpenGL.GL import *
from OpenGL.GLU import *

# ---------------------------------------------------------------
# 1. Configuracion
# ---------------------------------------------------------------
CARPETA_MODELOS = "modelos"
CARPETA_CONFIG = "ensamblaje"
ARCHIVO_CONFIG = "config_v4.json"
VERSION_CONFIG = 4
FPS = 30

# Nivel de cada pieza = cuantas articulaciones la mueven.
# Ejemplo: part3 es nivel 2 porque se mueve con J1 y con J2.
# La base (part1) es nivel 0 porque no se mueve.
NIVELES_PIEZAS = {
    "part1": 0,
    "part2": 1,
    "part3": 2,
    "part4": 3,
    "part5": 4,
    "part6": 5,
    "part7": 5,
    "part7a": 5,
}

# Las garras ademas de moverse con J1..J5 tienen su propia bisagra.
# El numero es el indice de la garra (0 o 1) para buscar su pivote.
PIEZAS_GARRA = {"part7": 0, "part7a": 1}

# Las dos garras giran en sentido contrario para abrir y cerrar
# (una con signo -1 y la otra con +1). Si en pantalla se ven al reves,
# basta con intercambiar los signos.
SIGNOS_GARRA = (-1.0, 1.0)

# Un color por pieza para distinguirlas en la pantalla
COLORES_PIEZAS = {
    "part1": (0.88, 0.48, 0.16),
    "part2": (0.20, 0.58, 0.92),
    "part3": (0.95, 0.78, 0.20),
    "part4": (0.20, 0.82, 0.48),
    "part5": (0.82, 0.35, 0.82),
    "part6": (0.95, 0.38, 0.25),
    "part7": (0.35, 0.75, 0.82),
    "part7a": (0.78, 0.78, 0.82),
}
COLOR_POR_DEFECTO = (0.70, 0.70, 0.70)

# Nombres para el menu de la pestana "Pivotes y ejes"
# (5 articulaciones + la bisagra de cada garra)
NOMBRES_JOINTS = [
    "J1 - Base (giro vertical)",
    "J2 - Hombro",
    "J3 - Codo",
    "J4 - Giro del tubo verde",
    "J5 - Muneca (rosa / terminal)",
    "Garra A - Part7 (bisagra)",
    "Garra B - Part7a (bisagra)",
]

# Deslizadores: las 5 articulaciones y la apertura de la garra
NOMBRES_SLIDERS = [
    "J1 - Base",
    "J2 - Hombro",
    "J3 - Codo",
    "J4 - Giro tubo verde",
    "J5 - Muneca",
    "Garra - Apertura",
]
# Limites de cada deslizador en grados (se pueden cambiar).
# La simulacion no detecta colisiones, por eso conviene limitarlos.
LIMITES_SLIDERS = [
    (-180, 180),
    (-90, 90),
    (-150, 150),
    (-180, 180),
    (-90, 90),
    (0, 40),
]
N_SLIDERS = len(NOMBRES_SLIDERS)   # 6 deslizadores
N_JOINTS = 5                       # articulaciones encadenadas J1..J5
N_MARCOS = len(NOMBRES_JOINTS)     # 7 (5 articulaciones + 2 bisagras)

# Pivotes y ejes medidos sobre los STL, en mm y en el sistema de
# coordenadas del ensamble de SolidWorks. Se calcularon ajustando
# circulos a los pasadores y agujeros de cada pieza:
#   J1: pasador entre Part1 y Part2, vertical
#   J2: pasador entre Part2 y Part3, eje Z
#   J3: pasador entre Part3 y Part4, eje Z
#   J4: eje del tubo verde (Part4), que coincide con el del conector de Part5
#   J5: pasador entre Part5 y Part6. En el ensamble quedo inclinado
#       unos 2.6 grados respecto a Z, y lo dejamos asi para que el giro
#       sea concentrico con los agujeros
#   Garras: pasadores de Part7 y Part7a sobre Part6 (mismo eje que J5)
PIVOTES_MEDIDOS = [
    [409.20, 102.10, 112.20],   # J1
    [259.10, 252.10, 100.00],   # J2
    [438.30, 492.65, 100.00],   # J3
    [534.47, 302.85, 139.00],   # J4 (un punto cualquiera sobre el eje)
    [629.64, 185.31, 137.93],   # J5
    [676.82, 150.58, 136.91],   # Garra A (Part7)
    [676.37, 220.56, 135.30],   # Garra B (Part7a)
]
EJES_MEDIDOS = [
    [0.0, 1.0, 0.0],                  # J1
    [0.0, 0.0, 1.0],                  # J2
    [0.0, 0.0, 1.0],                  # J3
    [0.6292, -0.7773, -0.0063],       # J4
    [0.0388, 0.0232, 0.9990],         # J5
    [0.0388, 0.0232, 0.9990],         # Garra A
    [0.0388, 0.0232, 0.9990],         # Garra B
]


# ---------------------------------------------------------------
# 2. Funciones matematicas y lectura de archivos
# ---------------------------------------------------------------
def normalizar(v):
    # Vuelve el vector de longitud 1 (vector unitario).
    # Si el vector es casi cero devolvemos el eje Z para no dividir por 0.
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def matriz_traslacion(vector):
    # Usamos matrices homogeneas de 4x4 para poder trasladar y rotar
    # con una sola multiplicacion. La traslacion va en la ultima columna.
    M = np.eye(4, dtype=np.float64)
    M[:3, 3] = np.asarray(vector, dtype=np.float64)
    return M


def matriz_rotacion_eje(eje, angulo):
    # Rotacion de un angulo (en grados) alrededor de un eje cualquiera
    # que pasa por el origen. Es la formula de Rodrigues:
    # R = I + sen(a)*K + (1 - cos(a))*K^2, desarrollada en cada termino.
    x, y, z = normalizar(eje)
    a = np.radians(float(angulo))
    c, s = np.cos(a), np.sin(a)
    C = 1.0 - c
    R = np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    return M


def transformar_punto(M, punto):
    # Aplica la matriz M a un punto 3D (se le agrega un 1 para que
    # funcione con la matriz de 4x4)
    p = np.array([punto[0], punto[1], punto[2], 1.0])
    return (M @ p)[:3]


def girar_alrededor(M, pivote_reposo, eje_reposo, angulo):
    # Rota alrededor de un pivote que no esta en el origen:
    #   1. llevamos el pivote al origen  -> T(-p)
    #   2. rotamos alrededor del eje     -> R
    #   3. devolvemos el pivote          -> T(p)
    # M es la transformacion acumulada de las articulaciones anteriores.
    # El pivote y el eje estan medidos con el brazo en reposo, entonces
    # primero hay que moverlos con M para saber donde estan ahora.
    pivote = transformar_punto(M, pivote_reposo)
    eje = normalizar(M[:3, :3] @ np.asarray(eje_reposo, dtype=np.float64))
    giro = (matriz_traslacion(pivote)
            @ matriz_rotacion_eje(eje, angulo)
            @ matriz_traslacion(-pivote))
    return giro @ M


def clave_pieza(nombre):
    # Saca el nombre corto de la pieza a partir del archivo.
    # Ejemplos: "Assem - Part7a-1.STL" -> "part7a", "Part3.stl" -> "part3"
    base = os.path.splitext(os.path.basename(nombre))[0].lower()
    m = re.search(r"part\s*(\d+)\s*(a?)", base)
    if not m:
        return None
    return "part" + m.group(1) + m.group(2)


def clave_natural(nombre):
    # Para ordenar "Part2" antes de "Part10" (orden natural y no alfabetico)
    partes = re.split(r"(\d+)", nombre.lower())
    return [int(p) if p.isdigit() else p for p in partes]


def leer_stl(ruta):
    # Lee un STL y devuelve los triangulos (N x 3 vertices x 3 coordenadas)
    # y la normal de cada triangulo.
    # Formato binario: 80 bytes de encabezado, 4 bytes con el numero de
    # triangulos y despues 50 bytes por triangulo (normal, 3 vertices y
    # 2 bytes de atributo).
    with open(ruta, "rb") as f:
        datos = f.read()
    if len(datos) >= 84:
        n = struct.unpack("<I", datos[80:84])[0]
        # Si el tamano coincide con la formula, el archivo es binario
        if 84 + n * 50 == len(datos):
            tipo = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            arr = np.frombuffer(datos, dtype=tipo, count=n, offset=84)
            tri = arr["v"].astype(np.float64)
            nor = arr["n"].astype(np.float64)
            return tri, _normales_validas(tri, nor)
    # Si no es binario se lee como texto (STL ASCII)
    texto = datos.decode("utf-8", errors="ignore")
    vs = re.findall(
        r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", texto)
    tri = np.array(vs, dtype=np.float64).reshape(-1, 3, 3)
    return tri, _normales_validas(tri, np.zeros((len(tri), 3)))


def _normales_validas(tri, nor):
    # Las normales sirven para la iluminacion. Usamos las que trae el
    # archivo y, si alguna viene en cero, la calculamos con el producto
    # cruz de dos lados del triangulo.
    calc = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(calc, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    calc = calc / ln
    lf = np.linalg.norm(nor, axis=1)
    ok = np.isfinite(lf) & (lf > 0.5)
    resultado = calc.copy()
    resultado[ok] = nor[ok] / lf[ok][:, None]
    return resultado


# ---------------------------------------------------------------
# 3. Visor 3D (la parte de OpenGL)
# ---------------------------------------------------------------
class Visor3D(OpenGLFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.piezas = []
        # Un angulo por deslizador (5 articulaciones + apertura de garra)
        self.angulos = [0.0] * N_SLIDERS
        # Se empieza con los pivotes y ejes medidos
        self.pivotes = [np.array(p, dtype=np.float64) for p in PIVOTES_MEDIDOS]
        self.ejes = [normalizar(e) for e in EJES_MEDIDOS]
        self.mostrar_ejes = True
        self.tamano_modelo = 500.0

        # Camara: gira alrededor de un centro, a una distancia (zoom)
        self.centro = np.zeros(3)
        self.zoom = 1500.0
        self.cam_x = 20.0      # elevacion
        self.cam_y = -35.0     # azimut

        self.gl_listo = False
        self.activo = True
        self.error_mostrado = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.bind("<ButtonPress-1>", self.mouse_inicio)
        self.bind("<B1-Motion>", self.mouse_mover)
        self.bind("<MouseWheel>", self.mouse_zoom)
        # En Linux la rueda del mouse llega como Button-4 y Button-5
        self.bind("<Button-4>", lambda e: self.cambiar_zoom(0.9))
        self.bind("<Button-5>", lambda e: self.cambiar_zoom(1.1))
        self.after(500, self.bucle_renderizado)

    def initgl(self):
        # Configuracion inicial de OpenGL (se hace una sola vez)
        glClearColor(0.12, 0.14, 0.18, 1.0)
        glEnable(GL_DEPTH_TEST)       # las piezas de adelante tapan a las de atras
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        # Con esto glColor tambien define el color del material iluminado
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        # Iluminar las dos caras del triangulo para que no queden
        # zonas negras si alguna normal apunta hacia adentro
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.28, 0.28, 0.28, 1.0])
        # LIGHT0 es la luz principal y LIGHT1 una luz de relleno mas suave
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.95, 0.95, 0.95, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.35, 0.35, 0.35, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.40, 0.42, 0.48, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        # Un poco de brillo para que se note la forma de las piezas
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.25, 0.25, 0.25, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 40.0)
        version = glGetString(GL_VERSION)
        if version:
            print("OpenGL activo:", version.decode(errors="replace"))

    def bucle_renderizado(self):
        # Se llama FPS veces por segundo: dibuja la escena y se
        # vuelve a programar con after()
        if not self.activo:
            return
        try:
            if not self.winfo_exists():
                return
            self.tkMakeCurrent()
            if not self.gl_listo:
                self.initgl()
                self.gl_listo = True
            self.redraw()
            if hasattr(self, "tkSwapBuffers"):
                self.tkSwapBuffers()
            else:
                glFlush()
            self.error_mostrado = False
        except Exception as error:
            # Se imprime el error una sola vez para no llenar la consola
            if not self.error_mostrado:
                print("Error de renderizado:", error)
                self.error_mostrado = True
        if self.activo:
            try:
                self.after(int(1000 / FPS), self.bucle_renderizado)
            except tk.TclError:
                self.activo = False

    # --- Cinematica ---
    def aplicar_joint(self, M, i):
        # Agrega a la transformacion acumulada M el giro de la articulacion i
        return girar_alrededor(M, self.pivotes[i], self.ejes[i], self.angulos[i])

    def matriz_pieza(self, pieza):
        # Transformacion final de una pieza. Se aplican las articulaciones
        # en orden, de J1 hasta la del nivel de la pieza, porque cada
        # eslabon depende de los anteriores (cadena cinematica).
        M = np.eye(4, dtype=np.float64)
        for i in range(min(pieza["nivel"], N_JOINTS)):
            M = self.aplicar_joint(M, i)
        # Las garras ademas giran en su propia bisagra
        g = pieza["garra"]
        if g is not None:
            M = girar_alrededor(
                M, self.pivotes[5 + g], self.ejes[5 + g],
                SIGNOS_GARRA[g] * self.angulos[5])
        return M

    def marcos_actuales(self):
        # Posicion actual de cada pivote y de cada eje, solo para dibujar
        # los marcadores (punto rojo y linea amarilla)
        M = np.eye(4, dtype=np.float64)
        marcos = []
        for i in range(N_JOINTS):
            marcos.append((transformar_punto(M, self.pivotes[i]),
                           normalizar(M[:3, :3] @ self.ejes[i])))
            M = self.aplicar_joint(M, i)
        # Despues de J5, M ya incluye las cinco articulaciones
        for g in range(2):
            marcos.append((transformar_punto(M, self.pivotes[5 + g]),
                           normalizar(M[:3, :3] @ self.ejes[5 + g])))
        return marcos

    # --- Dibujo ---
    def redraw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        ancho = max(self.winfo_width(), 1)
        alto = max(self.winfo_height(), 1)
        glViewport(0, 0, ancho, alto)

        # Camara en perspectiva. Los planos cercano y lejano dependen de
        # la distancia de la camara. Antes usabamos 0.1 y 100000 y las
        # superficies que se tocan parpadeaban (z-fighting) porque el
        # buffer de profundidad pierde precision con un rango tan grande.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, ancho / alto, self.zoom * 0.02, self.zoom * 12.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Las luces se ponen con la matriz identidad para que queden
        # fijas respecto a la camara (como una linterna)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.3, 0.6, 1.0, 0.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [-0.7, -0.3, -0.5, 0.0])

        # Posicion de la camara en coordenadas esfericas (angulos + distancia)
        elev = np.radians(self.cam_x)
        azim = np.radians(self.cam_y)
        ojo = self.centro + np.array([
            self.zoom * np.cos(elev) * np.sin(azim),
            self.zoom * np.sin(elev),
            self.zoom * np.cos(elev) * np.cos(azim),
        ])
        gluLookAt(*ojo, *self.centro, 0, 1, 0)

        self.dibujar_ejes_globales()
        for pieza in self.piezas:
            self.dibujar_vbo(pieza, self.matriz_pieza(pieza))
        if self.mostrar_ejes:
            self.dibujar_marcadores()

    def dibujar_ejes_globales(self):
        # Ejes del sistema de coordenadas del ensamble: X rojo, Y verde, Z azul
        largo = max(self.tamano_modelo * 0.15, 50.0)
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(largo, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, largo, 0)
        glColor3f(0.1, 0.35, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, largo)
        glEnd()
        glEnable(GL_LIGHTING)

    def dibujar_marcadores(self):
        # Punto rojo = pivote, linea amarilla = eje de giro.
        # Se dibujan sin prueba de profundidad para verlos aunque
        # esten dentro de una pieza. Sirven para comprobar que cada
        # pivote quedo en el centro de su articulacion.
        largo = max(self.tamano_modelo * 0.07, 15.0)
        marcos = self.marcos_actuales()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(3)
        glPointSize(10)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.85, 0.0)
        for punto, eje in marcos:
            glVertex3fv(np.asarray(punto - eje * largo, dtype=np.float32))
            glVertex3fv(np.asarray(punto + eje * largo, dtype=np.float32))
        glEnd()
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.0, 0.0)
        for punto, _ in marcos:
            glVertex3fv(np.asarray(punto, dtype=np.float32))
        glEnd()
        glLineWidth(1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # --- Carga de piezas y envio a la GPU ---
    def cargar_stl(self, ruta):
        nombre = os.path.basename(ruta)
        clave = clave_pieza(nombre)
        # Solo se cargan los archivos que sean parte del brazo
        if clave not in NIVELES_PIEZAS:
            print("Ignorado (no es una pieza del brazo):", nombre)
            return False
        if any(p["clave"] == clave for p in self.piezas):
            print("Ignorado (pieza repetida):", nombre)
            return False
        try:
            tri, nor = leer_stl(ruta)
            if len(tri) == 0:
                raise ValueError("El STL no contiene triangulos.")
            v = tri.reshape(-1, 3)
            self.piezas.append({
                "nombre": nombre,
                "clave": clave,
                "triangulos": tri.astype(np.float32),
                "normales": nor.astype(np.float32),
                # caja que encierra la pieza, se usa para centrar la camara
                "minimo": v.min(axis=0),
                "maximo": v.max(axis=0),
                "nivel": NIVELES_PIEZAS[clave],
                "garra": PIEZAS_GARRA.get(clave),
                "color": COLORES_PIEZAS.get(clave, COLOR_POR_DEFECTO),
                "vbo_vertices": None,
                "vbo_normales": None,
                "cantidad": 0,
            })
            print(f"STL cargado: {nombre} -> {clave} | nivel "
                  f"{NIVELES_PIEZAS[clave]} | triangulos: {len(tri)}")
            return True
        except Exception as error:
            print("Error cargando:", ruta, error)
            return False

    def preparar_gpu(self, pieza):
        # Se suben los vertices y normales a la memoria de la GPU una sola
        # vez (VBO). Despues, en cada cuadro, solo se manda a dibujar.
        # Es mucho mas rapido que enviar los triangulos uno por uno.
        if pieza["vbo_vertices"] is not None:
            return
        vertices = np.ascontiguousarray(
            pieza["triangulos"].reshape(-1, 3), dtype=np.float32)
        # Cada triangulo tiene una normal y se repite para sus 3 vertices
        normales = np.ascontiguousarray(
            np.repeat(pieza["normales"][:, None, :], 3, axis=1).reshape(-1, 3),
            dtype=np.float32)
        vbo_v = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_v)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        vbo_n = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_n)
        glBufferData(GL_ARRAY_BUFFER, normales.nbytes, normales, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        pieza["vbo_vertices"] = vbo_v
        pieza["vbo_normales"] = vbo_n
        pieza["cantidad"] = len(vertices)

    def dibujar_vbo(self, pieza, matriz):
        self.preparar_gpu(pieza)
        glPushMatrix()
        # OpenGL guarda las matrices por columnas, por eso se transpone
        glMultMatrixf(np.ascontiguousarray(matriz.T, dtype=np.float32))
        glColor3f(*pieza["color"])
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, pieza["vbo_vertices"])
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, pieza["vbo_normales"])
        glNormalPointer(GL_FLOAT, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, pieza["cantidad"])
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glPopMatrix()

    # --- Camara ---
    def ajustar_camara(self):
        # Centra la camara en el brazo y la aleja lo suficiente
        # para que se vea completo
        if not self.piezas:
            self.centro = np.zeros(3)
            self.zoom = 1500.0
            return
        minimo = np.min([p["minimo"] for p in self.piezas], axis=0)
        maximo = np.max([p["maximo"] for p in self.piezas], axis=0)
        self.centro = (minimo + maximo) / 2
        diagonal = float(np.linalg.norm(maximo - minimo))
        self.tamano_modelo = diagonal
        # El campo de vision es de 45 grados, por eso se usa sen(22.5)
        self.zoom = max(diagonal / 2 / np.sin(np.radians(22.5)) * 1.1, 10.0)

    def mouse_inicio(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

    def mouse_mover(self, event):
        # Arrastrar con el mouse gira la camara alrededor del brazo
        self.cam_y += (event.x - self.mouse_x) * 0.5
        self.cam_x = float(np.clip(
            self.cam_x + (event.y - self.mouse_y) * 0.5, -89, 89))
        self.mouse_x, self.mouse_y = event.x, event.y

    def cambiar_zoom(self, factor):
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 100000.0))

    def mouse_zoom(self, event):
        self.cambiar_zoom(0.9 if event.delta > 0 else 1.1)

    def home_camara(self):
        self.cam_x = 20.0
        self.cam_y = -35.0
        self.ajustar_camara()


# ---------------------------------------------------------------
# 4. Interfaz con Tkinter
# ---------------------------------------------------------------
class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador 3D - Brazo Robotico")
        self.root.geometry("1500x900")
        self.root.configure(bg="#202833")
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        # Las carpetas se buscan junto al archivo .py
        base = os.path.dirname(os.path.abspath(__file__))
        self.carpeta_modelos = os.path.join(base, CARPETA_MODELOS)
        self.carpeta_config = os.path.join(base, CARPETA_CONFIG)
        self.ruta_json = os.path.join(self.carpeta_config, ARCHIVO_CONFIG)

        self.sliders = []
        self.etiquetas = []
        self.vars_pivote = []
        self.vars_eje = []
        self.var_ejes = tk.BooleanVar(value=True)

        self.crear_interfaz()
        # Se esperan 900 ms para que la ventana de OpenGL ya exista
        # antes de cargar los modelos
        self.root.after(900, self.cargar_modelos)

    def crear_interfaz(self):
        # Panel izquierdo con controles y visor 3D a la derecha
        panel = tk.Frame(self.root, bg="#202833", width=400)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        panel.pack_propagate(False)

        tk.Label(panel, text="CONTROL DEL BRAZO ROBOTICO",
                 font=("Arial", 14, "bold"), bg="#202833",
                 fg="white").pack(pady=10)
        self.estado = tk.Label(panel, text="Buscando piezas STL...",
                               bg="#202833", fg="white", justify=tk.LEFT)
        self.estado.pack(pady=5)

        tabs = ttk.Notebook(panel)
        tabs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tab_mov = tk.Frame(tabs, bg="#202833")
        self.tab_piv = tk.Frame(tabs, bg="#202833")
        self.tab_piezas = tk.Frame(tabs, bg="#202833")
        tabs.add(self.tab_mov, text="Articulaciones")
        tabs.add(self.tab_piv, text="Pivotes y ejes")
        tabs.add(self.tab_piezas, text="Piezas")
        self.crear_tab_articulaciones()
        self.crear_tab_pivotes()
        self.crear_tab_piezas()

        botones = tk.Frame(panel, bg="#202833")
        botones.pack(fill=tk.X, padx=8, pady=5)
        for texto, comando in (("GUARDAR", self.guardar),
                               ("HOME ROBOT", self.home_robot),
                               ("HOME CAMARA", self.home_camara)):
            ttk.Button(botones, text=texto, command=comando).pack(
                side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.visor = Visor3D(self.root, width=1050, height=850)
        self.visor.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                        padx=8, pady=8)

    def crear_tab_articulaciones(self):
        # Un deslizador por cada movimiento. El indice i se pasa al
        # callback para saber cual angulo cambiar.
        tk.Label(self.tab_mov, text="MOVIMIENTOS", bg="#202833",
                 fg="#70B7FF", font=("Arial", 11, "bold")).pack(pady=10)
        for i, nombre in enumerate(NOMBRES_SLIDERS):
            minimo, maximo = LIMITES_SLIDERS[i]
            marco = tk.Frame(self.tab_mov, bg="#202833")
            marco.pack(fill=tk.X, padx=10, pady=5)
            fila = tk.Frame(marco, bg="#202833")
            fila.pack(fill=tk.X)
            tk.Label(fila, text=nombre, bg="#202833", fg="white",
                     font=("Arial", 9, "bold")).pack(side=tk.LEFT)
            etiqueta = tk.Label(fila, text="0°", bg="#202833", fg="#F4C76B")
            etiqueta.pack(side=tk.RIGHT)
            self.etiquetas.append(etiqueta)
            slider = tk.Scale(
                marco, from_=minimo, to=maximo, resolution=1,
                orient=tk.HORIZONTAL, length=340, bg="#202833", fg="white",
                troughcolor="#496783", highlightthickness=0,
                command=lambda valor, indice=i: self.cambiar_angulo(indice, valor))
            slider.pack(fill=tk.X)
            self.sliders.append(slider)
        tk.Checkbutton(
            self.tab_mov, text="Mostrar pivotes (rojo) y ejes (amarillo)",
            variable=self.var_ejes, command=self.alternar_ejes,
            bg="#202833", fg="white", selectcolor="#303D4D",
            activebackground="#202833", activeforeground="white"
        ).pack(padx=12, pady=8, anchor="w")

    def crear_campos_xyz(self, padre):
        # Tres cajas de texto (X, Y, Z). Devuelve las variables
        # donde queda lo que escribe el usuario.
        marco = tk.Frame(padre, bg="#202833")
        marco.pack(fill=tk.X, padx=5, pady=5)
        variables = []
        for eje in ("X", "Y", "Z"):
            sub = tk.Frame(marco, bg="#202833")
            sub.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
            tk.Label(sub, text=eje, bg="#202833", fg="white").pack()
            var = tk.StringVar(value="0")
            tk.Entry(sub, textvariable=var, width=10,
                     justify=tk.CENTER).pack()
            variables.append(var)
        return variables

    def crear_tab_pivotes(self):
        tk.Label(self.tab_piv, text="CENTRO Y DIRECCION DE GIRO",
                 bg="#202833", fg="#70B7FF",
                 font=("Arial", 10, "bold")).pack(pady=8)
        self.combo = ttk.Combobox(self.tab_piv, values=NOMBRES_JOINTS,
                                  state="readonly", width=34)
        self.combo.current(0)
        self.combo.pack(pady=5)
        self.combo.bind("<<ComboboxSelected>>",
                        lambda e: self.actualizar_campos_pivote())
        caja_pivote = tk.LabelFrame(self.tab_piv, text="Centro de giro (mm)",
                                    bg="#202833", fg="white")
        caja_pivote.pack(fill=tk.X, padx=8, pady=5)
        caja_eje = tk.LabelFrame(self.tab_piv, text="Direccion del eje",
                                 bg="#202833", fg="white")
        caja_eje.pack(fill=tk.X, padx=8, pady=5)
        self.vars_pivote = self.crear_campos_xyz(caja_pivote)
        self.vars_eje = self.crear_campos_xyz(caja_eje)
        ttk.Button(self.tab_piv, text="APLICAR CENTRO Y EJE",
                   command=self.aplicar_pivote).pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(self.tab_piv, text="RESTAURAR VALORES MEDIDOS",
                   command=self.restaurar_medidos).pack(fill=tk.X, padx=10, pady=5)
        tk.Label(
            self.tab_piv,
            text=("Los valores por defecto fueron medidos sobre\n"
                  "los STL del ensamble (coordenadas en mm).\n\n"
                  "Solo editalos si un pivote no queda en el\n"
                  "centro real de su union."),
            bg="#202833", fg="#cccccc", justify=tk.LEFT
        ).pack(padx=10, pady=10, anchor="w")

    def crear_tab_piezas(self):
        tk.Label(self.tab_piezas, text="PIEZAS DEL ROBOT", bg="#202833",
                 fg="#70B7FF", font=("Arial", 10, "bold")).pack(pady=8)
        self.lista_piezas = tk.Listbox(
            self.tab_piezas, height=18, bg="#303D4D", fg="white",
            selectbackground="#2878B5")
        self.lista_piezas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(
            self.tab_piezas,
            text=("Nivel 0: base fija\n"
                  "Nivel 1: se mueve con J1\n"
                  "Nivel 2: J1 + J2\n"
                  "Nivel 3: J1 + J2 + J3\n"
                  "Nivel 4: J1 a J4\n"
                  "Nivel 5: J1 a J5 (+ garra en Part7 y Part7a)"),
            bg="#202833", fg="#cccccc", justify=tk.LEFT
        ).pack(padx=10, pady=8)

    # --- Carga de modelos y configuracion ---
    def cargar_modelos(self):
        os.makedirs(self.carpeta_modelos, exist_ok=True)
        os.makedirs(self.carpeta_config, exist_ok=True)
        archivos = [a for a in os.listdir(self.carpeta_modelos)
                    if a.lower().endswith(".stl")]
        archivos.sort(key=clave_natural)
        for nombre in archivos:
            self.visor.cargar_stl(os.path.join(self.carpeta_modelos, nombre))

        if not self.visor.piezas:
            self.estado.config(
                text="No se encontraron STL en la carpeta 'modelos'")
            return

        self.leer_configuracion()
        self.actualizar_lista()
        self.actualizar_sliders()
        self.actualizar_campos_pivote()
        self.visor.ajustar_camara()

        # Se avisa si falta alguna de las 8 piezas
        cargadas = {p["clave"] for p in self.visor.piezas}
        faltan = sorted(set(NIVELES_PIEZAS) - cargadas, key=clave_natural)
        texto = f"Piezas cargadas: {len(cargadas)} de {len(NIVELES_PIEZAS)}"
        if faltan:
            texto += "\nFaltan: " + ", ".join(faltan)
            messagebox.showwarning(
                "Faltan piezas",
                "No se encontraron estas piezas en 'modelos':\n"
                + ", ".join(faltan))
        self.estado.config(text=texto)

    def leer_configuracion(self):
        # Si ya se habia guardado una configuracion, se recuperan los
        # angulos, pivotes y ejes. Solo se acepta la version actual del
        # archivo para no cargar datos de calibraciones anteriores.
        if not os.path.exists(self.ruta_json):
            return
        try:
            with open(self.ruta_json, "r", encoding="utf-8") as f:
                datos = json.load(f)
            if int(datos.get("version", 0)) != VERSION_CONFIG:
                return
            angulos = datos.get("angulos")
            if isinstance(angulos, list) and len(angulos) == N_SLIDERS:
                self.visor.angulos = [
                    float(np.clip(a, *LIMITES_SLIDERS[i]))
                    for i, a in enumerate(angulos)]
            pivotes = datos.get("pivotes")
            if isinstance(pivotes, list) and len(pivotes) == N_MARCOS:
                self.visor.pivotes = [np.asarray(p, dtype=np.float64)
                                      for p in pivotes]
            ejes = datos.get("ejes")
            if isinstance(ejes, list) and len(ejes) == N_MARCOS:
                self.visor.ejes = [normalizar(e) for e in ejes]
        except Exception as error:
            print("No se pudo leer la configuracion:", error)

    def actualizar_lista(self):
        self.lista_piezas.delete(0, tk.END)
        for p in self.visor.piezas:
            extra = " + garra" if p["garra"] is not None else ""
            self.lista_piezas.insert(
                tk.END, f"{p['nombre']} | {p['clave']} | Nivel {p['nivel']}{extra}")

    # --- Controles ---
    def cambiar_angulo(self, indice, valor):
        # Se ejecuta cada vez que se mueve un deslizador. Solo guarda
        # el angulo; el visor lo usa en el siguiente cuadro.
        angulo = float(valor)
        self.visor.angulos[indice] = angulo
        self.etiquetas[indice].config(text=f"{angulo:.0f}°")

    def actualizar_sliders(self):
        for i in range(N_SLIDERS):
            self.sliders[i].set(self.visor.angulos[i])
            self.etiquetas[i].config(text=f"{self.visor.angulos[i]:.0f}°")

    def alternar_ejes(self):
        self.visor.mostrar_ejes = bool(self.var_ejes.get())

    def actualizar_campos_pivote(self):
        # Muestra en las cajas de texto el pivote y eje de la
        # articulacion elegida en el menu
        i = self.combo.current()
        if i < 0:
            return
        for k in range(3):
            self.vars_pivote[k].set(f"{self.visor.pivotes[i][k]:.3f}")
            self.vars_eje[k].set(f"{self.visor.ejes[i][k]:.5f}")

    def aplicar_pivote(self):
        # Toma los numeros escritos y los usa como nuevo pivote y eje
        i = self.combo.current()
        if i < 0:
            return
        try:
            centro = np.array([float(v.get()) for v in self.vars_pivote])
            eje = np.array([float(v.get()) for v in self.vars_eje])
            if np.linalg.norm(eje) < 1e-12:
                raise ValueError("El eje no puede ser cero.")
            self.visor.pivotes[i] = centro
            self.visor.ejes[i] = normalizar(eje)
            self.actualizar_campos_pivote()
        except ValueError as error:
            messagebox.showerror("Error", str(error))

    def restaurar_medidos(self):
        self.visor.pivotes = [np.array(p, dtype=np.float64)
                              for p in PIVOTES_MEDIDOS]
        self.visor.ejes = [normalizar(e) for e in EJES_MEDIDOS]
        self.actualizar_campos_pivote()

    def guardar(self):
        datos = {
            "version": VERSION_CONFIG,
            "unidades": "mm (coordenadas del ensamble de SolidWorks)",
            "angulos": [float(a) for a in self.visor.angulos],
            "pivotes": [p.tolist() for p in self.visor.pivotes],
            "ejes": [e.tolist() for e in self.visor.ejes],
        }
        try:
            os.makedirs(self.carpeta_config, exist_ok=True)
            with open(self.ruta_json, "w", encoding="utf-8") as f:
                json.dump(datos, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Guardado",
                                "Se guardaron angulos, pivotes y ejes.")
        except Exception as error:
            messagebox.showerror("Error al guardar", str(error))

    def home_robot(self):
        # Vuelve a la posicion del ensamble (todos los angulos en 0)
        self.visor.angulos = [0.0] * N_SLIDERS
        self.actualizar_sliders()

    def home_camara(self):
        self.visor.home_camara()

    def cerrar(self):
        # Antes de cerrar se liberan los buffers de la GPU
        self.visor.activo = False
        try:
            self.visor.tkMakeCurrent()
            for p in self.visor.piezas:
                buffers = [b for b in (p["vbo_vertices"], p["vbo_normales"])
                           if b is not None]
                if buffers:
                    glDeleteBuffers(len(buffers), buffers)
        except Exception as error:
            print("Aviso al liberar GPU:", error)
        self.root.destroy()


# ---------------------------------------------------------------
# 5. Programa principal
# ---------------------------------------------------------------
def main():
    root = tk.Tk()
    Aplicacion(root)
    root.mainloop()


if __name__ == "__main__":
    main()