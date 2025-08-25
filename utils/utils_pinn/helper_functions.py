import numpy as np
from scipy.optimize import newton
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# Funktion berechnet Höhe eines Kreissegments auf Basis des Kreisradius r und der Fläche A
def getHeight(A, r):
    eq = lambda h: A - r**2 * np.arccos(1 - h / r) + (r - h) * np.sqrt(2 * r * h - h**2)
    h0 = r / 2
    if A < 0:
        #print('Querschnitt kleiner Null: ' + str(A))
        return 0
    elif A > np.pi * r**2:
        #print('Querschnitt größer als zulässig: ' + str(A))
        return 2*r
    return newton(eq, h0)

def getHeightArray(A, r):
    h = np.zeros_like(A)
    for i in range(len(h)):
        h[i] = getHeight(A[i], r)
    return h

# Funktion berechnet die Fläche eines Kreissegments auf Basis des Kreisradiuses r und der Höhe h des Segments
def getArea(h, r):
    return r**2 * np.arccos(1 - h / r) - (r - h) * np.sqrt(2 * r * h - h**2)

def yaron(eta_c, eta_d, eps, eta_v=23e-3):
    al = eta_c / (eta_d + eta_v)
    ga = eps ** (1 / 3)
    omega = ((4 * ga ** 7 + 10 - (84 / 11) * ga ** 2 + 4 * al * (1 - ga ** 7)) /
             (10 * (1 - ga ** 10) - 25 * ga ** 3 * (1 - ga ** 4) + 10 * al * (1 - ga ** 3) * (1 - ga ** 7)))
    return eta_c * (1 + 5.5 * omega * eps)


def calc_efficiency(d_j, N_j):
    V_end = 0
    V_0 = 0
    for j in range(len(d_j)):
        V_end += (np.pi/6)*(d_j[j]**3)*N_j[-1,j]
        V_0 += (np.pi/6)*(d_j[j]**3)*N_j[0,j]
    return 1 - (V_end/V_0)
    