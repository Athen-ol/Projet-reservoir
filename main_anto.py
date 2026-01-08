import reservoirpy
from reservoirpy import ESN
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.hyper import research, plot_hyperopt_report, parallel_research
from reservoirpy.nodes import Reservoir, Ridge
from dataset_prep import init_dataset
import numpy as np
import matplotlib.pyplot as plt

reservoirpy.set_seed(96)

nb_training_points = 8000
nb_test_points = 100
pas = 10


SET_A, SET_B, SET_C = init_dataset("dst_labels.csv")

# fig, axs = plt.subplots(3, 1, figsize=(10, 9))
# axs[0].set_title("Preview of the normalized DST signal")
# axs[0].plot(SET_A)
# axs[0].set_ylabel("$jsp$")
# axs[0].set_xlabel("$t$")
# axs[1].plot(SET_B)
# axs[1].set_ylabel("$jsp$")
# axs[1].set_xlabel("$t$")
# axs[2].plot(SET_C)
# axs[2].set_ylabel("$jsp$")
# axs[2].set_xlabel("$t$")
# plt.show()

# dataset = to_forecasting(SET_A[0:nb_training_points], forecast=10)
x_train = SET_A[0:nb_training_points]
y_train = SET_A[pas:nb_training_points+pas]
x_test = SET_A[nb_training_points:nb_training_points+nb_test_points]

# plt.figure(figsize=(10, 3))
# plt.title("Preview of the normalized DST signal")
# plt.ylabel("$jsp$")
# plt.xlabel("$t$")
# plt.plot(y_train, label="train", color="blue")
# plt.plot(x_train, label="test", color="red")
# plt.show()

model = ESN(units=500, 
            sr=0.9, 
            lr=0.5,
            ridge=1e-6, 
            input_scaling=2.0)

model = model.fit(x_train, y_train)
y_test = model.run(x_test)

plt.figure(figsize=(10, 3))
plt.title("ESN prediction on the normalized DST signal")
plt.ylabel("$jsp$")
plt.xlabel("$t$")
plt.plot(x_test, label="train", color="blue")
plt.plot(y_test, label="pred", color="red")
plt.legend()
plt.show()

# pas1 = 1

# x_train, y_train = SET_A[0:nb_training_points], SET_A[pas1:nb_training_points+pas1]
# x_test = SET_A[nb_training_points:nb_training_points+nb_test_points]

# plt.figure(figsize=(10, 3))
# plt.title("Preview of the normalized DST signal")
# plt.ylabel("$jsp$")
# plt.xlabel("$t$")
# plt.plot(y_train, label="train", color="blue")
# plt.plot(x_train, label="test", color="red")
# plt.show()

# model1 = ESN(units=100, 
#             sr=0.9, 
#             lr=0.5,
#             ridge=1e-6, 
#             input_scaling=1.0)

# model1 = model1.fit(x_train, y_train)
# # y_test = model1.run(x_test)

# y_test = np.empty((pas, 1))
# x = y_train[-1]

# for i in range(pas):
#     x = model1(x)
#     y_test[i] = x

# plt.figure(figsize=(10, 3))
# plt.title("ESN prediction on the normalized DST signal")
# plt.ylabel("$jsp$")
# plt.xlabel("$t$")
# plt.plot(x_test, label="train", color="blue")
# plt.plot(y_test, label="pred", color="red")
# plt.legend()
# plt.show()


def objective(dataset, config, *, input_scaling, N, sr, lr, ridge, seed):
    # This step may vary depending on what you put inside 'dataset'
    x_train, x_test, y_train, y_test = dataset

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []; r2s = [];
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(
            units=N,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            seed=variable_seed
        )

        readout = Ridge(ridge=ridge)

        model = reservoir >> readout


        # Train your model and test your model.
        predictions = model.fit(x_train, y_train) \
                           .run(x_test)

        loss = nrmse(y_test, predictions, norm_value=np.ptp(x_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}

import json

# hyperopt_config = {
#     "exp": "hyperopt-multiscroll",    # the experimentation name
#     "hp_max_evals": 200,              # the number of differents sets of parameters hyperopt has to try
#     "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
#     "seed": 96,                       # the random state seed, to ensure reproducibility
#     "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
#     "hp_space": {                     # what are the ranges of parameters explored
#         "N": ["choice", 500],             # the number of neurons is fixed to 500
#         "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
#         "lr": ["loguniform", 1e-3, 1],    # idem with the leaking rate, from 1e-3 to 1
#         "input_scaling": ["choice", 1.0], # the input scaling is fixed
#         "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
#         "seed": ["choice", 80085]          # an other random seed for the ESN initialization
#     }
# }

# # we precautionously save the configuration in a JSON file
# # each file will begin with a number corresponding to the current experimentation run number.
# with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
#     json.dump(hyperopt_config, f)

# nb_training_points = 5000
# nb_test_points = 2000
# pas = 10

# X_train = SET_A[0:nb_training_points]
# Y_train = SET_A[pas : nb_training_points + pas]
# # X_test = SET_A[nb_training_points : - pas]
# X_test = SET_A[nb_training_points : nb_training_points + nb_test_points]
# Y_test = SET_A[nb_training_points + pas : nb_training_points + pas + nb_test_points]    
# dataset = (X_train, X_test, Y_train, Y_test)

# best = parallel_research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

# fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr", "ridge"), metric="r2")
# plt.show()

# hyperopt_config = {
#     "exp": "hyperopt-multiscroll",    # the experimentation name
#     "hp_max_evals": 200,              # the number of differents sets of parameters hyperopt has to try
#     "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
#     "seed": 96,                       # the random state seed, to ensure reproducibility
#     "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
#     "hp_space": {                     # what are the ranges of parameters explored
#         "N": ["choice", 500],             # the number of neurons is fixed to 500
#         "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
#         "lr": ["loguniform", 1e-1, 1],    # idem with the leaking rate, from 1e-3 to 1
#         "input_scaling": ["choice", 1.0], # the input scaling is fixed
#         "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
#         "seed": ["choice", 80085]          # an other random seed for the ESN initialization
#     }
# }

# # we precautionously save the configuration in a JSON file
# # each file will begin with a number corresponding to the current experimentation run number.
# with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
#     json.dump(hyperopt_config, f)

# nb_training_points = 5000
# nb_test_points = 2000
# pas = 10

# X_train = SET_A[0:nb_training_points]
# Y_train = SET_A[pas : nb_training_points + pas]
# # X_test = SET_A[nb_training_points : - pas]
# X_test = SET_A[nb_training_points : nb_training_points + nb_test_points]
# Y_test = SET_A[nb_training_points + pas : nb_training_points + pas + nb_test_points]    
# dataset = (X_train, X_test, Y_train, Y_test)

# best = parallel_research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

# fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr", "ridge"), metric="r2")
# plt.show()

model = ESN(units=1000, 
            sr=0.5, 
            lr=1.0,
            ridge=1e-5, 
            input_scaling=1.0)

model = model.fit(x_train, y_train)
y_test = model.run(x_test)

plt.figure(figsize=(10, 3))
plt.title("ESN prediction on the normalized DST signal")
plt.ylabel("$jsp$")
plt.xlabel("$t$")
plt.plot(x_test, label="train", color="blue")
plt.plot(y_test, label="pred", color="red")
plt.legend()
plt.show()

# hyperopt_config = {
#     "exp": "hyperopt-multiscroll",    # the experimentation name
#     "hp_max_evals": 50,              # the number of differents sets of parameters hyperopt has to try
#     "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
#     "seed": 96,                       # the random state seed, to ensure reproducibility
#     "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
#     "hp_space": {                     # what are the ranges of parameters explored
#         "N": ["choice", 2000],             # the number of neurons is fixed to 500
#         "sr": ["choice", 0.5],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
#         "lr": ["choice", 1.0],    # idem with the leaking rate, from 1e-3 to 1
#         "input_scaling": ["choice", 1.0], # the input scaling is fixed
#         "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
#         "seed": ["choice", 80085]          # an other random seed for the ESN initialization
#     }
# }

hyperopt_config = {
    "exp": "hyperopt-multiscroll",    # the experimentation name
    "hp_max_evals": 100,              # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
    "seed": 96,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
    "hp_space": {                     # what are the ranges of parameters explored
        "N": ["choice", 1000],             # the number of neurons is fixed to 500
        "sr": ["choice", 0.5],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "lr": ["choice", 0.99],    # idem with the leaking rate, from 1e-3 to 1
        "input_scaling": ["choice", 1.0], # the input scaling is fixed
        "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
        "seed": ["choice", 80085]          # an other random seed for the ESN initialization
    }
}

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

nb_training_points = 5000
nb_test_points = 2000
pas = 10

X_train = SET_A[0:nb_training_points]
Y_train = SET_A[pas : nb_training_points + pas]
# X_test = SET_A[nb_training_points : - pas]
X_test = SET_A[nb_training_points : nb_training_points + nb_test_points]
Y_test = SET_A[nb_training_points + pas : nb_training_points + pas + nb_test_points]    
dataset = (X_train, X_test, Y_train, Y_test)

best = parallel_research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr", "ridge"), metric="r2")
plt.show()