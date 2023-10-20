import matplotlib.pyplot as plt
import numpy as np
import math


def differentiate(series, time):
    diff = np.zeros(series.shape)
    for i, data in enumerate(series):
        if i == 0:
            continue
        diff[i] = (series[i] - series[i - 1]) / (time[i] - time[i - 1])

    return diff

# constant r strategy
r10 = np.load("servoing_20_0.05_2.npy", "r")
r20 = np.load("servoing_20_0.1_6.npy", "r")
r30 = np.load("servoing_20_0.2_2.npy", "r")
r40 = np.load("servoing_20_0.3_2.npy", "r")

fig = plt.figure(figsize=(24, 12))
ax1 = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(2, 2, 1)
# ax3 = fig.add_subplot(2, 2, 2)

x, y, z, theta0, psi, phi, vx, vy, vz, q, r, p = r10[0:12]
time0 = r10[12]
divergence0 = r10[13]
distance0 = r10[14]
divergence_dot0 = differentiate(divergence0, time0)

x, y, z, theta1, psi, phi, vx, vy, vz1, q1, r, p = r20[0:12]
time1 = r20[12]
divergence1 = r20[13]
distance1 = r20[14]
acceleration = differentiate(vz1, time1)
acceleration_est = (9.81) * theta1 - vz1
divergence_dot1 = differentiate(divergence1, time1)
distance_est1 = -acceleration / (divergence1*divergence1 - divergence_dot1)  # r20[15]

x, y, z, theta2, psi, phi, vx, vy, vz, q, r, p = r30[0:12]
time2 = r30[12]
divergence2 = r30[13]
distance2 = r30[14]
divergence_dot2 = differentiate(divergence2, time2)

x, y, z, theta3, psi, phi, vx, vy, vz, q, r, p = r40[0:12]
time3 = r40[12]
divergence3 = r40[13]
distance3 = r40[14]
divergence_dot3 = differentiate(divergence3, time3)


# states - 0:12, time - 12, divergence - 13, distances - 14
ax1.set_title("Constant divergence approach with step-wise increasing set-points based on estimated distance, Delay=0.1s, Noise: Var = 0.0001")
# ax1.plot(time1, distance1, label='True Distance')
# ax1.plot(time1, 40*divergence_dot1, label='Divergence derivative', linestyle='--')
# ax1.plot(time1, distance_est1, label='Distance Estimate', linestyle='--')
ax1.plot(time1, acceleration, label='Acceleration', linestyle='--')
ax1.plot(time1, acceleration_est, label='Acceleration Estimate', linestyle='--')
# ax1.plot(time1, (theta1 * 17.2 * divergence1**(-0.808)), label='estimate')
# ax1.plot(time1, divergence1*40, label='Divergence * 40')
# # ax1.plot(time1, divergence_dot1*40, label='Divergence Dot * 40')
# ax1.plot(time1, theta1, label='theta')
# ax1.plot(time1, gain1, label='confidence')
# ax1.plot(time1, np.ones(len(time1))*2, label='confidence')


ax1.set_ylim([-3, 20])
# ax1.set_xlim([-12, 0])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Distance [m]')
# ax1.set_ylabel('Distance [m]')
ax1.legend()
ax1.grid()

plt.show()

