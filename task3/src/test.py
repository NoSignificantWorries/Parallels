import matplotlib.pyplot as plt
import numpy as np


# Sin results
with open("results/sin.txt") as file:
    sin_x = []
    sin_y = []
    for line in file.readlines()[1:]:
        line = line[:-1]
        line = line.split(" ")[2:]
        x, y = tuple(map(lambda x: round(float(x), 3), line))
        sin_x.append(x)
        sin_y.append(y)

sin_x = np.array(sin_x)
sin_y = np.array(sin_y)

sin_sorter = np.argsort(sin_x)
sin_x = sin_x[sin_sorter]
sin_y = sin_y[sin_sorter]

X_sin = np.linspace(sin_x.min(), sin_x.max(), sin_x.size)
Y_sin = np.sin(X_sin)


# Sqrt results
with open("results/sqrt.txt") as file:
    sqrt_x = []
    sqrt_y = []
    for line in file.readlines()[1:]:
        line = line[:-1]
        line = line.split(" ")[2:]
        x, y = tuple(map(lambda x: round(float(x), 3), line))
        sqrt_x.append(x)
        sqrt_y.append(y)

sqrt_x = np.array(sqrt_x)
sqrt_y = np.array(sqrt_y)

sqrt_sorter = np.argsort(sqrt_x)
sqrt_x = sqrt_x[sqrt_sorter]
sqrt_y = sqrt_y[sqrt_sorter]

X_sqrt = np.linspace(sqrt_x.min(), sqrt_x.max(), sqrt_x.size)
Y_sqrt = np.sqrt(X_sqrt)

# Pow results
with open("results/pow.txt") as file:
    list_pow_x = []
    list_pow_y = []
    list_pow_z = []
    for line in file.readlines()[1:]:
        line = line[:-1]
        line = list(line.split(" ")[2:])
        line[0] = line[0][1:]
        line[1] = line[1][:-1]
        x, y, z = tuple(map(lambda x: round(float(x), 3), line))
        list_pow_x.append(x)
        list_pow_y.append(y)
        list_pow_z.append(z)

pow_x = np.array(list_pow_x)
pow_y = np.array(list_pow_y)
pow_z = np.array(list_pow_z)

np_z = np.array([np.pow(x, y) for x, y in zip(pow_x, pow_y)])

pow_sorter = np.argsort(pow_x)
pow_x = pow_x[pow_sorter]
pow_z = pow_z[pow_sorter]
np_z = np_z[pow_sorter]

_, axises = plt.subplots(nrows=1, ncols=3, figsize=(25, 12))

axises[0].plot(X_sin, Y_sin, label="True", linewidth=4)
axises[0].plot(sin_x, sin_y, "-o", label="From threads")
axises[0].set_title("Sin")
axises[0].legend()

axises[1].plot(X_sqrt, Y_sqrt, label="True", linewidth=4)
axises[1].plot(sqrt_x, sqrt_y, "-o", label="From threads")
axises[1].set_title("Sqrt")
axises[1].legend()

axises[2].plot(pow_x, np_z, label="True", linewidth=4)
axises[2].plot(pow_x, pow_z, "-o", label="From threads")
axises[2].set_title("Pow")
axises[2].legend()

plt.savefig("results/main.png", format="png", dpi=600)
