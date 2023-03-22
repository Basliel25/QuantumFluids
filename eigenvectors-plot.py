import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1
m = 1
L = 10
N = 100
dx = L/N
x = np.linspace(-L/2, L/2, N)

# External potential
V_ext = 0.5 * m * x**2

# Hartree potential
rho = np.ones(N)
V_H = np.zeros(N)
for i in range(N):
    V_H[i] = np.trapz(rho*np.abs(x[i]-x)/np.sqrt((x[i]-x)**2 + dx**2), x)

# Exchange-correlation potential
V_xc = np.zeros(N)
rs = (3/4/np.pi/rho)**(1/3)
eps_xc = np.zeros(N)
for i in range(N):
    eps_xc[i] = -3/4*(3/np.pi)**(1/3)/rs[i]
V_xc = eps_xc*rho

# Total potential
V = V_ext + V_H + V_xc

# Kinetic energy operator
T = -(hbar**2/2/m)*(1/dx**2)*(-2*np.diag(np.ones(N)) + np.diag(np.ones(N-1), -1) + np.diag(np.ones(N-1), 1))

# Hamiltonian
H = T + np.diag(V)

# Solve eigenvalue problem
E, psi = np.linalg.eigh(H)

# Plot wave functions
for j in range(5):
    plt.plot(x, psi[:, j], label=f"$E_{j}$={E[j]:.2f}")
plt.xlabel("x")
plt.ylabel("$\psi(x)$")
plt.legend()
plt.show()

