import matplotlib.pyplot as plt

with open("result_integration.txt", "r") as file:
   data = [line[1:-2].split(" ") for line in file.readlines()]
   data[0] = list(map(int, data[0]))
   data[1] = list(map(float, data[1]))
   data[2] = list(map(int, data[2]))
   data[3] = list(map(float, data[3]))
   data[4] = list(map(int, data[4]))
   data[5] = list(map(float, data[5]))

threads_counts = [1, 2, 4, 7, 8, 16, 20, 40]

plt.title("Parallel integration")

plt.grid(True)
plt.xlabel("p")
plt.ylabel("Sp")

plt.plot(threads_counts, data[1], label="sin")
plt.plot(threads_counts, data[3], label="exp")
plt.plot(threads_counts, data[5], label="sqrt")
plt.plot(threads_counts, threads_counts, "g--", label="Linear")

plt.legend()

plt.savefig('result_integration.png', format='png', dpi=600)

plt.show()

