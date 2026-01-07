import reservoirpy as rpy
import test_reservoir as tres
from reservoirpy.nodes import Reservoir
from reservoirpy.datasets import to_forecasting
import numpy as np
import matplotlib.pyplot as plt

rpy.set_seed(96)

nb_points = 200
pas = 1


SET_A, SET_B, SET_C = tres.init_dataset("dst_labels.csv")

fig, axs = plt.subplots(3, 1, figsize=(10, 9))
axs[0].set_title("Preview of the normalized DST signal")
axs[0].set_ylabel("$jsp$")
axs[0].set_xlabel("$t$")
axs[0].plot(SET_A)
axs[1].plot(SET_B)
axs[1].set_ylabel("$jsp$")
axs[1].set_xlabel("$t$")
axs[2].plot(SET_C)
axs[2].set_ylabel("$jsp$")
axs[2].set_xlabel("$t$")
plt.show()

# dataset = to_forecasting(SET_A[0:200], forecast=10)
x_train, y_train = SET_A[0:200], SET_A[pas:200+pas]
y_test = SET_A[200:250]

plt.figure(figsize=(10, 3))
plt.title("Preview of the normalized DST signal")
plt.ylabel("$jsp$")
plt.xlabel("$t$")
plt.plot(y_train, label="train", color="blue")
plt.plot(x_train, label="test", color="red")
plt.show()

model = rpy.ESN(units=100, 
            sr=0.9, 
            lr=0.5,
            ridge=1e-6, 
            input_scaling=1.0)

model = model.fit(x_train, y_train)
y_pred = model.run(y_test)

plt.figure(figsize=(10, 3))
plt.title("ESN prediction on the normalized DST signal")
plt.ylabel("$jsp$")
plt.xlabel("$t$")
plt.plot(y_test, label="train", color="blue")
plt.plot(y_pred, label="pred", color="red")
plt.legend()
plt.show()

