import numpy as np


def syssim(A, B, T, N, i, sigu, sigw, sigx):
    n = A[i].shape[0]  # Nombre de lignes de A[i]
    p = B[i].shape[1]  # Nombre de colonnes de B[i]

    # x = np.zeros((n, T+1))
    # u = np.zeros((p, T))
    # w = np.zeros((n, T))
    # z = np.zeros((n + p, T))
    X = np.zeros((n, T * N))
    Z = np.zeros((n + p, T * N))
    W = np.zeros((n, T * N))
    list_x0 = []

    for j in range(N):

        x = np.zeros((n, T+1))
        u = np.zeros((p, T))
        w = np.zeros((n, T))
        z = np.zeros((n + p, T))
     
        # Initialisation de l'état avec une distribution normale
        x0 = np.random.normal(0, sigx, n)
        list_x0.append(x0)
        x[:, 0] = x0
    
        for k in range(T):
            # Génération des entrées et bruits
            u[:, k] = np.random.normal(0, sigu, (p,))
            w[:, k] = np.random.normal(0, sigw, (n,))
            z[:, k] = np.concatenate([x[:, k], u[:, k]])
            
            x[:, k + 1] = A[i] @ x[:, k] + B[i] @ u[:, k] + w[:, k]

        # Exclusion du premier état
        x = x[:, 1:]
        # Inversion de l'ordre des colonnes
        x = np.fliplr(x)
        z = np.fliplr(z)
        w = np.fliplr(w)

        # Stockage des résultats dans les matrices de sortie
        X[:, j * T:(j + 1) * T] = x
        Z[:, j * T:(j + 1) * T] = z
        W[:, j * T:(j + 1) * T] = w

    return X, Z, W, list_x0
