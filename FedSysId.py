import numpy as np
import matplotlib.pyplot as plt

from syssim import syssim 
from sysgen import sysgen
from sysid import sysid



# System Data
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

# Noise levels
sigu = 1
sigw = 1
sigx = 1

# Rollout length
T = 5

# Select FL_solver (Fed_Avg = 0, Fed_Lin = 1)
FL_solver = 0

q = 25  # number of estimations
R = 200  # number of global iterations
s = 1  # fixed system for the error computation

# Simulation 1: Varying M
M = [1, 2, 5, 25, 100]
N = 25  # Fixed number of rollouts
epsilon = 0.01  # Fixed dissimilarity

# Generate system matrices
# sysgen is a function that you need to implement
A, B = sysgen(A_0, B_0, V, U, M, epsilon)  # Generate similar systems

E_avg = np.zeros((len(M), R))

# Numerical results varying the number of clients
Error_matrix = np.zeros((q, R))
for j in range(len(M)):
    for i in range(q):
        # sysid is a function that you need to implement
        Error_matrix[i, :] = sysid(A, B, T, N, M[j], R, sigu, sigx, sigw, FL_solver, s)
    E_avg[j, :] = np.mean(Error_matrix, axis=0)

S1 = E_avg

# Simulation 2: Varying N
M = 50  # Fixed number of clients
N = [5, 25, 50, 75, 100]  # Number of rollouts

# Numerical results varying the number of rollouts
for j in range(len(N)):
    for i in range(q):
        Error_matrix[i, :] = sysid(A, B, T, N[j], M, R, sigu, sigx, sigw, FL_solver, s)
    E_avg[j, :] = np.mean(Error_matrix, axis=0)

S2 = E_avg

# Simulation 3: Varying delta
M = 50  # Fixed number of clients
N = 25  # Fixed number of rollouts
epsilon_values = [0.01, 0.1, 0.25, 0.5, 0.75]  # Dissimilarity values

# Numerical results for different epsilon values
for j in range(len(epsilon_values)):
    A, B = sysgen(A_0, B_0, V, U, M, epsilon_values[j])
    for i in range(q):
        Error_matrix[i, :] = sysid(A, B, T, N, M, R, sigu, sigx, sigw, FL_solver, s)
    E_avg[j, :] = np.mean(Error_matrix, axis=0)

S3 = E_avg

# Plotting the results

title_fig = 'FedAvg' if FL_solver == 0 else 'FedLin'

# Error vs number of global iterations - varying M
plt.figure(1)
plt.hold(True)
for i in range(len(M)):
    plt.plot(np.arange(1, R + 1), S1[i, :], linewidth=1.2)
plt.hold(False)
plt.legend([f'M={m}' for m in M], interpreter='latex')
plt.xlabel('r', fontsize=20)
plt.ylabel('$e_r$', fontsize=20)
plt.xlim([1, R])
plt.grid(True)
plt.title(title_fig)

# Error vs number of global iterations - varying N
plt.figure(2)
N_values = [5, 25, 50, 75, 100]
for i in range(len(N_values)):
    plt.plot(np.arange(1, R + 1), S2[i, :], linewidth=1.2)
plt.legend([f'$N_i={n}$' for n in N_values], interpreter='latex')
plt.xlabel('r', fontsize=20)
plt.ylabel('$e_r$', fontsize=20)
plt.xlim([1, R])
plt.grid(True)
plt.title(title_fig)

# Error vs number of global iterations - varying epsilon
plt.figure(3)
epsilon_values = [0.01, 0.1, 0.25, 0.5, 0.75]
for i in range(len(epsilon_values)):
    plt.plot(np.arange(1, R + 1), S3[i, :], linewidth=1.2)
plt.legend([f'$\epsilon={e}$' for e in epsilon_values], interpreter='latex')
plt.xlabel('r', fontsize=20)
plt.ylabel('$e_r$', fontsize=20)
plt.xlim([1, R])
plt.grid(True)
plt.title(title_fig)

plt.show()
