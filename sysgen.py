import numpy as np


def generate_gamma_pairs(M, epsilon):
    """
    Génère M paires (γ1, γ2) selon U(0, ε) où M est le nombre de clients

    Parameters:
        n (int): Nombre de paires à générer.
        epsilon (float): Valeur maximale de l'intervalle [0, ε].

    Returns:
        np.ndarray: Tableau de forme (M, 2) contenant les paires générées.
    """
    return np.random.uniform(0, epsilon, size=(M, 2))


def generate_A_B(M, epsilon, A0, B0, V, U):
    gamma_pairs = generate_gamma_pairs(M, epsilon)

    # Créer les listes pour A et B
    A_list = [A0 + gamma_pairs[i, 0] * V for i in range(M)]
    B_list = [B0 + gamma_pairs[i, 1] * U for i in range(M)]

    return A_list, B_list


# Juste histoire de reprendre les mêmes notations que le git de l'artcile
def sysgen(A_0, B_0, V, U, M, epsilon):
    """
    Génère les systèmes A et B en fonction des paramètres donnés.
    Si M est un entier, génère M systèmes.
    Si M est une liste, utilise la valeur maximale de M pour générer les systèmes.
    """
    if isinstance(M, int):  # Si M est un entier
        A, B = generate_A_B(M, epsilon, A_0, B_0, V, U)
    elif isinstance(M, list):  # Si M est une liste
        A, B = generate_A_B(max(M), epsilon, A_0, B_0, V, U)
    else:
        raise ValueError("M doit être un entier ou une liste.")

    return A, B