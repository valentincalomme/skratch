import numpy as np
import matplotlib.pyplot as plt

num_points = 50
f = lambda x: np.sin(x)
x = np.linspace(-10, 10, num_points)
y = f(x) + np.random.normal(0,0.5, len(x)) # function of the curve with some normal noise added


plt.figure(figsize=(20,10))

plt.subplot(3,1,1)

data, = plt.plot(x, y, '.', markersize=10)
good, = plt.plot(x, f(x), 'g')
plt.legend([data, good, high_bias, high_variance], ["Noisy Data", "Denoised Data"])

plt.subplot(3,1,2)

data, = plt.plot(x, y, '.', markersize=10)
high_bias, = plt.plot(x, np.zeros_like(x),  'b')
plt.legend([data, good, high_bias, high_variance], ["Noisy Data", "High bias"])

plt.subplot(3,1,3)

data, = plt.plot(x, y, '.', markersize=10)
high_variance, = plt.plot(x, y, 'r')
plt.legend([data, good, high_bias, high_variance], ["Noisy Data", "High variance"])


plt.show()
