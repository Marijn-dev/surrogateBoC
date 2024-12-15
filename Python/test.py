import numpy as np
from scipy.integrate import solve_ivp

# Define the ODE
def exponential_decay(t, y):
    return -y

# Initial conditions
t_span = (0, 5)
y0 = [1]

# Solve the ODE
solution = solve_ivp(exponential_decay, t_span, y0, method='RK45')

# Extract and plot the solution
import matplotlib.pyplot as plt
plt.plot(solution.t, solution.y[0], label='y(t)')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Exponential Decay')
plt.legend()
plt.show()

print(solution.y)