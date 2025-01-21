import numpy as np
import matplotlib.pyplot as plt

from sysgen import sysgen
from sysid import sysid

# System Data:

# System dimensions
n = 3
p = 2

# Nominal system
A_0 = np.array([[0.6, 0.5, 0.4],
                [0, 0.4, 0.3],
                [0, 0, 0.3]])

B_0 = np.array([[1, 0.5],
                [0.5, 1],
                [0.5, 0.5]])

V = np.array([[0, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])  # Modification pattern applied to A_0
U = np.array([[1, 0],
              [0, 0],
              [0, 1]])  # Modification pattern applied to B_0

true_theta = np.hstack([A_0, B_0])



# Noise level, input signal, and initial state
sigu = 1
sigw = 1
sigx = 1

# Rollout length
T = 5

# Selecting the FL_solver: Fed_Avg (FL_solver = 0) and Fed_Lin (FL_solver = 1)
FL_solver = 0

q = 25  # Number of estimations
R = 200  # Number of global iterations
s = 0  # Fixed system for the error computation

# Simulation 1: varying M
# M = [1, 2, 5, 25, 100]  # Number of clients
M = [1,1]
N = 25  # Fixed number of rollouts
epsilon = 0.01  # Fixed dissimilarity

# Generating the system matrices
# Define `sysgen` to generate similar systems
A, B = sysgen(A_0, B_0, V, U, M, epsilon)

E_avg = np.zeros((len(M), R))

# Numerical results varying the number of clients
Error_matrix = np.zeros((q, R))



for j in range(len(M)):
    # print(j)
    # print(M[j])  
    for i in range(q):
        print("\n")
        tmp = sysid(A, B, T, N, M[j], R, sigu, sigx, sigw, FL_solver, s, true_theta)
        print(tmp, "\n")
        Error_matrix[i, :] = tmp
    E_avg[j, :] = np.mean(Error_matrix, axis=0)

S1 = E_avg
print(S1.shape)


# Création du graphique
plt.figure(figsize=(10, 6))
for i, m in enumerate(M):
    plt.plot(range(R), S1[i, :], label=f'M = {m} clients')

# Ajout des légendes et des titres
plt.xlabel('r')  # Nom de l'abscisse
plt.ylabel('$e_r$')  # Nom de l'ordonnée
plt.title("Évolution de l'erreur en fonction des rounds pour différents nombres de clients")
plt.legend(title="Nombre de clients")
plt.grid(True)

# Sauvegarde de l'image
plt.savefig("S1_M_variation.png", dpi=300, bbox_inches='tight')
plt.close()  # Ferme la figure pour libérer la mémoire






# # Simulation 2: varying N
# M = 50  # Fixed number of clients
# N = [5, 25, 50, 75, 100]  # Number of rollouts

# for j, n in enumerate(N):
#     for i in range(q):
#         Error_matrix[i, :] = sysid(A, B, T, n, M, R, sigu, sigx, sigw, FL_solver, s)
#     E_avg[j, :] = np.mean(Error_matrix, axis=0)

# S2 = E_avg

# # Simulation 3: varying epsilon
# M = 50  # Fixed number of clients
# N = 25  # Fixed number of rollouts
# epsilon = [0.01, 0.1, 0.25, 0.5, 0.75]  # Dissimilarity levels

# for j, e in enumerate(epsilon):
#     A, B = sysgen(A_0, B_0, V, U, M, e)
#     for i in range(q):
#         Error_matrix[i, :] = sysid(A, B, T, N, M, R, sigu, sigx, sigw, FL_solver, s)
#     E_avg[j, :] = np.mean(Error_matrix, axis=0)

# S3 = E_avg

# # Illustrating the numerical results
# title_fig = 'FedAvg' if FL_solver == 0 else 'FedLin'

# # Error vs number of global iterations - varying M
# plt.figure(1)
# for i, m in enumerate(M):
#     plt.plot(range(1, R + 1), S1[i, :], label=f'M={m}', linewidth=1.2)
# plt.legend()
# plt.xlabel('r')
# plt.ylabel('$e_r$')
# plt.title(title_fig)
# plt.grid()
# plt.show()

# # Error vs number of global iterations - varying N
# plt.figure(2)
# for i, n in enumerate(N):
#     plt.plot(range(1, R + 1), S2[i, :], label=f'N={n}', linewidth=1.2)
# plt.legend()
# plt.xlabel('r')
# plt.ylabel('$e_r$')
# plt.title(title_fig)
# plt.grid()
# plt.show()

# # Error vs number of global iterations - varying epsilon
# plt.figure(3)
# for i, e in enumerate(epsilon):
#     plt.plot(range(1, R + 1), S3[i, :], label=f'epsilon={e}', linewidth=1.2)
# plt.legend()
# plt.xlabel('r')
# plt.ylabel('$e_r$')
# plt.title(title_fig)
# plt.grid()
# plt.show()
