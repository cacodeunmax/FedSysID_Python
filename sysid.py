import numpy as np
from syssim import syssim  # Importation de la fonction syssim depuis syssim.py

def sysid(A, B, T, N, M, R, sigu, sigx, sigw, FL_solver, s, true_theta=None):
    n = A[0].shape[0]
    p = B[0].shape[1]

    # Générer les trajectoires du système
    X = []
    Z = []
    W = []
    for i in range(M):
        X_i, Z_i, W_i, list_x0 = syssim(A, B, T, N, i, sigu, sigw, sigx)
        X.append(X_i)
        Z.append(Z_i)
        W.append(W_i)

    # FedSysID

    # Initialiser le serveur avec \bar{\Theta}_0 et \alpha
    Theta_0 = np.hstack([(1/2) * A[s], (1/2) * B[s]])
    alpha = 1e-4  # pas d'apprentissage
    K = 10  # nombre d'itérations locales
    mu = 2.5

    Theta_s = Theta_0.copy()  # serveur
    Theta_c = [None] * M  # clients
    Error = np.zeros(R)  # vecteur d'erreur  de dim (R,)

    for r in range(R):
        # Initialiser chaque client avec \bar{\Theta}_0
        for i in range(M):
            Theta_c[i] = Theta_s.copy()

        # Côté client :
        for i in range(M):
            if FL_solver == 1:
                # FedLin
                g = np.zeros((n, n + p))
                for s in range(M):
                    g -= (X[s] - Theta_s @ Z[s]) @ Z[s].T
                g /= M

                for k in range(K):
                    Theta_c[i] -= alpha * (-(X[i] - Theta_c[i] @ Z[i]) @ Z[i].T +
                                           (X[i] - Theta_s @ Z[i]) @ Z[i].T + g)

            if FL_solver == 0:
                # FedAvg
                for k in range(1, K + 1):
                    new_val = Theta_c[i] + (alpha / k) * ((X[i] - Theta_c[i] @ Z[i]) @ Z[i].T)
                    Theta_c[i] = new_val
            
            if FL_solver == 2:
                # FedAvg
                for k in range(1, K + 1):
                    new_val = Theta_c[i] + (alpha / k)*(((X[i] - Theta_c[i] @ Z[i]) @ Z[i].T) - mu*(Theta_c[i] - Theta_s))
                    Theta_c[i] = new_val

        # Côté serveur :
        Theta_sum = np.zeros((n, n + p))
        for i in range(M):
            Theta_sum = Theta_sum + Theta_c[i]
        Theta_s = (1 / M) * Theta_sum

        # Calcul de l'erreur en fonction de true_theta
        if true_theta is not None:
            Error[r] = np.linalg.norm(Theta_s - true_theta, ord=2)  # Erreur avec true_theta
        else:
            Error[r] = np.linalg.norm(Theta_s - np.hstack([A[s], B[s]]), ord=2)  # Erreur avec la valeur de référence

    return Error



# import numpy as np
# from syssim import syssim  # Importation de la fonction syssim depuis syssim.py


# def sysid(A, B, T, N, M, R, sigu, sigx, sigw, FL_solver, s):
#     n = A[0].shape[0]
#     p = B[0].shape[1]

#     # Générer les trajectoires du système
#     X = []
#     Z = []
#     W = []
#     for i in range(M):
#         X_i, Z_i, W_i, list_x0 = syssim(A, B, T, N, i, sigu, sigw, sigx)
#         X.append(X_i)
#         Z.append(Z_i)
#         W.append(W_i)

#     # FedSysID

#     # Initialiser le serveur avec \bar{\Theta}_0 et \alpha
#     Theta_0 = np.hstack([(1/2) * A[s], (1/2) * B[s]])
#     alpha = 1e-4  # pas d'apprentissage
#     K = 10  # nombre d'itérations locales


#     Theta_s = Theta_0.copy()  # serveur
#     Theta_c = [None] * M  # clients
#     Error = np.zeros(R) # vecteur d'erreur  de dim (R,)

#     for r in range(R):
#         # Initialiser chaque client avec \bar{\Theta}_0
#         for i in range(M):
#             Theta_c[i] = Theta_s.copy()

#         # Côté client :
#         for i in range(M):
#             if FL_solver == 1:
#                 # FedLin
#                 g = np.zeros((n, n + p))
#                 for s in range(M):
#                     g -= (X[s] - Theta_s @ Z[s]) @ Z[s].T
#                 g /= M

#                 for k in range(K):
#                     Theta_c[i] -= alpha * (-(X[i] - Theta_c[i] @ Z[i]) @ Z[i].T +
#                                            (X[i] - Theta_s @ Z[i]) @ Z[i].T + g)

#             elif FL_solver == 0:
#                 # FedAvg
#                 for k in range(1, K + 1):
#                     Theta_c[i] = Theta_c[i] + (alpha / k) * ((X[i] - Theta_c[i] @ Z[i]) @ Z[i].T)

#         # Côté serveur :
#         Theta_sum = np.zeros((n, n + p))
#         for i in range(M):
#             Theta_sum = Theta_sum + Theta_c[i]
#         Theta_s = (1 / M) * Theta_sum
#         Error[r] = np.linalg.norm(Theta_s - np.hstack([A[s], B[s]]),ord=2) # TODO : askip ici l'erreur c la norme 2 et jsp pk  # DONE : ord=2 impose norme spectale

#     return Error
