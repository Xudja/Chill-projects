import numpy as np
import matplotlib.pyplot as plt
import cv2
from functools import lru_cache

lru_cache(None)
scale_percent = 500

# Initialize physical and numerical constants
L = 1.0  # Set the length of the system
dx = 0.01  # Set the discrete spatial stepsize
c = 1  # Define the wave speed

dt = 0.707* dx / c  # Choose a time step to satisfy the CFL condition

x = np.arange(0, L * (1 + dx), dx)  # Define an array to store x position data
y = np.arange(0, L * (1 + dx), dx)  # Define an array to store y position data

xx, yy = np.meshgrid(x, y)

nsteps = 1 # Set the number of time steps

f = np.zeros((len(x), len(y), 3))
g = np.zeros((len(x), len(y)))
g = g.astype('float')


xc = 0.5  # Define the center of the system to locate a Gaussian pulse
w = 0.05  # Define the width of the Gaussian wave pulse
t = 0
f[:, :, 0] = np.exp(-(xx - xc) ** 2 / (w ** 2)) * np.exp(-(yy - xc) ** 2 / (w ** 2))  # Use Gaussian initial condition
# print(f)
# First time step in the leap frog algorithm
f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + 0.5 * c ** 2 * (
        f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2. * f[1:-1, 1:-1, 0]) * (dt ** 2 / dx ** 2) + 0.5 * c ** 2 * (
                           f[1:-1, :-2, 0] + f[1:-1, 2:, 0] - 2. * f[1:-1, 1:-1, 0]) * (dt ** 2 / dx ** 2)\


while True:
    t+= dx/c
    f[len(x)//2, len(y)//2, 1] = 10*np.sin(2*np.pi/t)
    # For all additional time steps
    for k in range(nsteps):
        f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2 * f[1:-1, 1:-1, 1] + c ** 2 * (
                f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2. * f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2) + c ** 2 * (
                                   f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2. * f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2)

        # Push the data back for the leapfrogging
        f[:, :, 0] = f[:, :, 1]
        f[:, :, 1] = f[:, :, 2]


    f_g = np.copy(f[:, :, 2])
    # print(f_g)

    g += (abs((f_g.astype('float'))))**(1/3)
    # print(g)

    # print(f[:, :, 2])
    # print('------------------------------------------------------')
    #
    # print(f)
    # print('------------------------------------------------------')
    frame = f[:, :, 2]
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dsize = (width, height)
    frame = cv2.resize(f, dsize)
    # width_g = int(frame_g.shape[1] * scale_percent / 100)
    # height_g = int(frame_g.shape[0] * scale_percent / 100)
    # dsize_g = (width_g, height_g)
    # frame_g = cv2.resize(frame_g, dsize)

    cv2.imshow('mask', frame)
    # cv2.imshow('mask_g', frame_g)

    k = cv2.waitKey(1) and 0xFF
    if k == 27:
        break
    if t>1000:
        break

def f(x,max):
    return x/max

max = np.max(g)
print(max,'0000000000')
s = g/np.max(g)
# s = s.astype('uint8')
print(s)
width_g = int(s.shape[1] * scale_percent / 100)
height_g = int(s.shape[0] * scale_percent / 100)
dsize_g = (width_g, height_g)
g = cv2.resize(s, dsize)
cv2.imshow('mask1', g)


cv2.waitKey(0)
cv2.destroyAllWindows()
