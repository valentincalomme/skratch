import numpy as np
import matplotlib.pyplot as plt


num_points = 50

x = np.linspace(-10, 10, num_points)
x_noise = x + np.random.normal(0, 0.7, len(x))


f1 = lambda x: x**1
f2 = lambda x: x**2
f3 = lambda x: x**3
f4 = lambda x: x**4
y1 = f1(x_noise)  # function of the curve with some normal noise added
y2 = f2(x_noise)  # function of the curve with some normal noise added
y3 = f3(x_noise)  # function of the curve with some normal noise added
y4 = f4(x_noise)  # function of the curve with some normal noise added

plt.figure(figsize=(20, 10))

for y, f, s in zip([y1, y2, y3, y4], [f1, f2, f3, f4], [221, 222, 223, 224]):

    plt.subplot(s)
    data, = plt.plot(x, y, 'Xr', markersize=10)
    fitted, = plt.plot(x, f(x), 'b', markersize=10)
    plt.legend([data, fitted], ["Data", "Fitted curve"])

plt.show()
