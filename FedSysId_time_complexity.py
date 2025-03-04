import numpy as np
import matplotlib.pyplot as plt

from sysgen import sysgen
from sysid import sysid
from sysid_time import sysid_time


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



q = 25  # Number of estimations
R = 200  # Number of global iterations
s = 0  # Fixed system for the error computation



M = 50
N = 25  # Fixed number of rollouts
epsilon = 0.01  # Fixed dissimilarity

# Generating the system matrices
# Define `sysgen` to generate similar systems
A, B = sysgen(A_0, B_0, V, U, M, epsilon)

Error, time_taken  = sysid_time(A, B, T, n, M, R, sigu, sigx, sigw, 0, s)


FL_solver_list = [0, 1, 2]
FL_solver_list_names =[ "FedAvg","FedLin","FedProx"]


Time_matrix = np.zeros((len(FL_solver_list), R))


for i in range(len(FL_solver_list)):
    FL_solver= FL_solver_list[i]
    Error, time_taken  = sysid_time(A, B, T, n, M, R, sigu, sigx, sigw, FL_solver, s)
    Time_matrix[i, :] = time_taken
    print(f"FL_solver {i} done.")
S4 = Time_matrix




###############################################
import numpy as np
import matplotlib.pyplot as plt

# Calcul de la somme des temps pour chaque solver
cumulative_times = np.cumsum(Time_matrix, axis=1)  # Cumulative sum along each row (solver)

# Création d'une figure avec deux sous-graphes côte à côte
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Premier graphique : Temps par round
for i in range(len(FL_solver_list)):
    solver_name = FL_solver_list_names[i] if i < len(FL_solver_list_names) else f"FL_solver {i}"
    ax[0].plot(range(R), Time_matrix[i, :], label=solver_name)

# Ajouter les labels et le titre pour le premier graphique
ax[0].set_xlabel('Round (r)', fontsize=12)
ax[0].set_ylabel('Time (seconds)', fontsize=12)
ax[0].set_title('Time Taken for Different FL Solvers', fontsize=14)
ax[0].grid(True)

# Deuxième graphique : Somme des temps par round
for i in range(len(FL_solver_list)):
    solver_name = FL_solver_list_names[i] if i < len(FL_solver_list_names) else f"FL_solver {i}"
    ax[1].plot(range(R), cumulative_times[i, :], label=solver_name)

# Ajouter les labels et le titre pour le deuxième graphique
ax[1].set_xlabel('Round (r)', fontsize=12)
ax[1].set_ylabel('Cumulative Time (seconds)', fontsize=12)
# ax[1].set_title('Cumulative Time for Different FL Solvers', fontsize=14)
ax[1].grid(True)

# Ajouter la légende
ax[0].legend()
ax[1].legend()

# Enregistrer l'image sous le nom 'S4_time_per_rounds_cumulative.png'
plt.tight_layout()  # Ajuster l'espacement entre les graphes
plt.savefig('S4_time_per_rounds_cumulative.png', format='png')


###############################################

# Calcul du temps moyen d'exécution par round pour chaque solver
mean_times = np.mean(Time_matrix, axis=1)

# Calcul du temps total d'exécution pour chaque solver (somme sur tous les rounds)
total_times = np.sum(Time_matrix, axis=1)

# Création d'une figure avec deux sous-graphes côte à côte
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Premier graphique : Temps moyen par round (nuage de points)
ax[0].scatter(FL_solver_list_names, mean_times, color='blue', s=100, label='Mean Time')
ax[0].set_xlabel('FL Solvers', fontsize=12)
ax[0].set_ylabel('Average Time (seconds)', fontsize=12)
ax[0].set_title('Average Time Per Round for Different FL Solvers', fontsize=14)
ax[0].grid(True)

# Deuxième graphique : Temps total d'exécution (nuage de points)
ax[1].scatter(FL_solver_list_names, total_times, color='red', s=100, label='Total Time')
ax[1].set_xlabel('FL Solvers', fontsize=12)
ax[1].set_ylabel('Total Time (seconds)', fontsize=12)
ax[1].set_title('Total Time for All Rounds for Different FL Solvers', fontsize=14)
ax[1].grid(True)

# Enregistrer l'image sous le nom 'S4_mean_total_round.png'
plt.tight_layout()  # Ajuster l'espacement entre les graphes
plt.savefig('S4_mean_total_round.png', format='png')



######



plt.figure(figsize=(10, 6))

# Plot cumulative times for each solver
for i in range(len(FL_solver_list)):
    solver_name = FL_solver_list_names[i] if i < len(FL_solver_list_names) else f"FL_solver {i}"
    plt.plot(range(R), cumulative_times[i, :], label=solver_name)

# Add labels and title
plt.xlabel('Round (r)', fontsize=12)
plt.ylabel('Cumulative Time (seconds)', fontsize=12)
# plt.title('Cumulative Time for Different FL Solvers', fontsize=14)
plt.grid(True)
plt.legend()

# Save the plot as an image
plt.tight_layout()
plt.savefig('S4_cumulative_time_per_round.png', format='png')
plt.show()