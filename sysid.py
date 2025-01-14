import numpy as np
from syssim import syssim  # Importation de la fonction syssim depuis syssim.py


def sysid(A, B, T, N, M, R, sigu, sigx, sigw, FL_solver, s):
    # Dimensions
    n = A[0].shape[0]  # Taille de l'état
    p = B[0].shape[1]  # Taille de l'entrée

    # Génération des trajectoires du système
    X = {}
    Z = {}
    W = {}

    for i in range(M):
        X[i], Z[i], W[i] = syssim(A, B, T, N, i, sigu, sigw, sigx)

    # Initialisation de l'identification fédérée
    Theta_0 = np.hstack([(1/2) * A[s], (1/2) * B[s]])  # Initialisation de Theta_0
    alpha = 1e-4  # Taille du pas
    K = 10  # Nombre d'itérations locales

    Theta_s = Theta_0  # Paramètres du serveur
    Theta_c = {}  # Paramètres des clients
    Error = np.zeros(R)  # Erreurs

    for l in range(R):
        # Initialisation de chaque client avec Theta_s
        for i in range(M):
            Theta_c[i] = Theta_s

        # Partie client
        for i in range(M):
            if FL_solver == 1:  # FedLin
                g = np.zeros((n, n + p))
                for s in range(M):
                    g -= (X[s] - Theta_s @ Z[s]) @ Z[s].T
                g = (1/M) * g

                for k in range(K):
                    Theta_c[i] -= alpha * (-(X[i] - Theta_c[i] @ Z[i]) @ Z[i].T + (X[i] - Theta_s @ Z[i]) @ Z[i].T + g)

            elif FL_solver == 0:  # FedAvg
                for k in range(K):
                    Theta_c[i] += (alpha / k) * ((X[i] - Theta_c[i] @ Z[i]) @ Z[i].T)

        # Partie serveur
        Theta_sum = np.zeros((n, n + p))
        for i in range(M):
            Theta_sum += Theta_c[i]
        Theta_s = (1/M) * Theta_sum

        # Calcul de l'erreur
        Error[l] = np.linalg.norm(Theta_s - np.hstack([A[s], B[s]]))

    return Error
