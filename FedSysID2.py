import numpy as np
import matplotlib.pyplot as plt

from sysgen import sysgen
from sysid import sysid


# System Data:

# System dimensions
n = 3
p = 2

# Nominal system:
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

# Noise level, input signal, and initial state
sigu = 1
sigw = 1
sigx = 1

# Rollout length:
T = 5

# Selecting the FL_solver: Fed_Avg (FL_solver = 0) and Fed_Lin (FL_solver = 1)
FL_solver = 0

q = 25  # Number of estimations
R = 200  # Number of global iterations
s = 1  # Fixed system for the error computation

# Simulation 1: varying M

# Number of clients
M = [1, 2, 5, 25, 100]

# Fixed number of rollouts:
N = 25

# Fixed dissimilarity
epsilon = 0.01

# Generating the system matrices:
# Function `sysgen` must be defined elsewhere.
A, B = sysgen(A_0, B_0, V, U, M, epsilon)  # Generates similar systems

E_avg = np.zeros((len(M), R))

# Numerical results varying the number of clients participating in the collaboration
Error_matrix = np.zeros((q, R))

for j, m in enumerate(M):
    for i in range(q):
        Error_matrix[i, :] = sysid(A, B, T, N, m, R, sigu, sigx, sigw, FL_solver, s)
    E_avg[j, :] = np.mean(Error_matrix, axis=0)

S1 = E_avg

# Simulation 2: varying N

# Fixed number of clients:
M = 50

# Number of rollouts:
N = [5, 25, 50, 75, 100]

# Here we set epsilon = 0.01
E_avg = np.zeros((len(N), R))

for j, n in enumerate(N):
    for i in range(q):
        Error_matrix[i, :] = sysid(A, B, T, n, M, R, sigu, sigx, sigw, FL_solver, s)
    E_avg[j, :] = np.mean(Error_matrix, axis=0)

S2 = E_avg

# Simulation 3: varying delta

# Fixed number of clients:
M = 50

# Fixed number of rollouts:
N = 25

# Dissimilarity
epsilon = [0.01, 0.1, 0.25, 0.5, 0.75]

E_avg = np.zeros((len(epsilon), R))

for j, eps in enumerate(epsilon):
    A, B = sysgen(A_0, B_0, V, U, M, eps)
    for i in range(q):
        Error_matrix[i, :] = sysid(A, B, T, N, M, R, sigu, sigx, sigw, FL_solver, s)
    E_avg[j, :] = np.mean(Error_matrix, axis=0)

S3 = E_avg

# Illustrating the numerical results:

title_fig = 'FedAvg' if FL_solver == 0 else 'FedLin'

# Error vs number of global iterations - varying M
plt.figure(1)
colors = ['b', 'g', 'r', 'c', 'k']
for i, m in enumerate(M):
    plt.plot(range(1, R + 1), S1[i, :], label=f'M={m}', color=colors[i], linewidth=1.2)
plt.xlabel('r')
plt.ylabel('$e_r$')
plt.title(title_fig)
plt.legend()
plt.grid(True)
plt.show()

# Error vs number of global iterations - varying N
plt.figure(2)
colors = ['b', 'g', 'r', 'c', 'm']
for i, n in enumerate(N):
    plt.plot(range(1, R + 1), S2[i, :], label=f'N={n}', color=colors[i], linewidth=1.2)
plt.xlabel('r')
plt.ylabel('$e_r$')
plt.title(title_fig)
plt.legend()
plt.grid(True)
plt.show()

# Error vs number of global iterations - varying epsilon
plt.figure(3)
colors = ['b', 'g', 'r', 'c', 'm']
for i, eps in enumerate(epsilon):
    plt.plot(range(1, R + 1), S3[i, :], label=f'epsilon={eps}', color=colors[i], linewidth=1.2)
plt.xlabel('r')
plt.ylabel('$e_r$')
plt.title(title_fig)
plt.legend()
plt.grid(True)
plt.show()
