import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.set_title("")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()

x = np.linspace(-10, 10, 100)
y = np.cos(3*x) + np.sin(2*x)

ax.plot(x,y)

plt.show()















































