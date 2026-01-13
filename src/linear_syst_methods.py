# -*- coding: utf-8 -*-
"""
Python 3
05 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np

# ####################################################################
def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante eliminación gaussiana."""
    mult_div = 0
    sum_rest = 0
    intercambios = 0

    # CAMBIO 1: Copiamos A para no modificar la original
    A = np.array(A, copy=True, dtype=float) 
    
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(0, n - 1):
        # --- encontrar pivote
        p = None
        for pi in range(i, n):
            if A[pi, i] == 0: continue
            if p is None:
                p = pi
                continue
            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            intercambios += 1
            logging.debug(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            mult_div += 1
            k = (n + 1) - i
            mult_div += k  # para la multiplicación m * A[i, i:]
            sum_rest += k  # para la resta A[j, i:] - ...
            A[j, i:] = A[j, i:] - m * A[i, i:]

        logging.info(f"\n{A}")

    # CAMBIO 2: Corregir indentación. Antes el 'if' y el 'print' estaban mal anidados.
    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")
        
    # --- Sustitución hacia atrás
    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
            mult_div += 1
            sum_rest += 1
        solucion[i] = (A[i, n] - suma) / A[i, i]
        sum_rest += 1
        mult_div += 1

    return solucion, mult_div, sum_rest, intercambios


# ####################################################################
def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A.
    [IMPORTANTE] No se realiza pivoteo.

    ## Parameters

    ``A``: matriz cuadrada de tamaño n-by-n.

    ## Return

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior. Se obtiene de la matriz ``A`` después de aplicar la eliminación gaussiana.
    """

    A = np.array(
        A, dtype=float
    )  # convertir en float, porque si no, puede convertir como entero

    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]

    L = np.zeros((n, n), dtype=float)
    P = np.identity(n, dtype=float)


    for i in range(0, n):  # loop por columna

        # --- deterimnar pivote
        pivot_index = np.argmax(np.abs(A[i:, i])) + i
        if pivot_index != i:
            P[[i, pivot_index]] = P[[pivot_index, i]]
            A[[i, pivot_index]] = A[[pivot_index, i]]
            if i > 0:
                L[[i, pivot_index], :i] = L[[pivot_index, i], :i]
        
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")

        # --- Eliminación: loop por fila
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

            L[j, i] = m

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    return P, L, A



def resolver_LU(P: np.ndarray,L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema mediante descomposición LU."""
    n = L.shape[0]
    
    # CAMBIO: Aplanar b para evitar problemas de (n,1) vs (n,)
    b = P @ b 
    
    # --- Sustitución hacia adelante
    y = np.zeros(n, dtype=float)
    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        suma = 0
        for j in range(i):
            suma += L[i, j] * y[j]
        y[i] = (b[i] - suma) / L[i, i]

    # --- Sustitución hacia atrás
    sol = np.zeros(n, dtype=float)
    sol[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += U[i, j] * sol[j]
        sol[i] = (y[i] - suma) / U[i, i]

    return sol


# ####################################################################
def matriz_aumentada(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Construye la matriz aumentada de un sistema de ecuaciones lineales.

    ## Parameters

    ``A``: matriz de coeficientes.

    ``b``: vector de términos independientes.

    ## Return

    ``a``:

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    assert A.shape[0] == b.shape[0], "Las dimensiones de A y b no coinciden."
    return np.hstack((A, b.reshape(-1, 1)))


# ####################################################################
def separar_m_aumentada(Ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa la matriz aumentada en la matriz de coeficientes y el vector de términos independientes.

    ## Parameters
    ``Ab``: matriz aumentada.

    ## Return
    ``A``: matriz de coeficientes.
    ``b``: vector de términos independientes.
    """
    logging.debug(f"Ab = \n{Ab}")
    if not isinstance(Ab, np.ndarray):
        logging.debug("Convirtiendo Ab a numpy array")
        Ab = np.array(Ab, dtype=float)
    return Ab[:, :-1], Ab[:, -1].reshape(-1, 1)
