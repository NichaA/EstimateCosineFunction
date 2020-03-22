# Estimate the cosine function
import numpy as np
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

# Assign number of available data points and resolution for X
N = 10000
theta_resolution = 0.01

# create the training data set
theta = np.arange(-180, 180 + theta_resolution, theta_resolution)
X_train = np.random.choice(theta, N)
y_train = np.cos(X_train * np.pi / 180) * (1 + np.random.random_integers(-1000, 1000, X_train.shape) / 10000)

# normalize X to [-1,1]
X_train /= 180.

# split the data into training and test sets (not in use)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_test = theta / 180
y_test = np.cos(X_test * np.pi)


# define the model
def create_model(hidden_layers=0, neurons=8, reg_lambda=0.01):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=1, activation='relu', kernel_regularizer=regularizers.l2(reg_lambda)))
    for i in range(hidden_layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(reg_lambda)))
    model.add(Dense(1))
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


# create model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=100, verbose=1)

# define the grid search parameters
hidden_layers = [1, 2, 4, 8]
neurons = [16, 32, 64]
reg_lambda = [0.1, 0.01, 0.001]
batch_size = [100, 200, 400]
epochs = [50, 100]
param_space = dict(hidden_layers=hidden_layers, neurons=neurons, reg_lambda=reg_lambda,
                   batch_size=batch_size, epochs=epochs)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_space, scoring='neg_mean_squared_error',
                          n_jobs=1, cv=3, n_iter=50)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# get the optimized model and evaluate with test set
optimized_model = grid.best_estimator_

# make predictions with the model
predictions = np.squeeze(optimized_model.predict(X_test))

# display the first 10 cases
for i in range(10):
    print('%s => %f (expected %f)' % (X_test[i].tolist(), predictions[i], y_test[i]))

# evaluate the model
testmse = mean_squared_error(y_test, predictions)
print('MSE: %.5f' % (testmse))

# observe the results
plt.scatter(X_test*180, predictions)
plt.plot(X_test*180, np.cos(X_test*np.pi), '-r')
plt.show()

# observe the residuals with the Bland-Altman plot
differences = predictions - y_test

mintheta = np.min(theta)
maxtheta = np.max(theta)
vertical_margin = 0.002
plt.scatter(X_test*180, differences)
plt.hlines(np.mean(differences), mintheta, maxtheta)
plt.hlines(np.mean(differences)+1.96*np.std(differences), mintheta, maxtheta, linestyles='dashed')
plt.hlines(np.mean(differences)-1.96*np.std(differences), mintheta, maxtheta, linestyles='dashed')
plt.text(0.5*maxtheta, np.mean(differences)+vertical_margin, "Mean = %0.05f" % np.mean(differences))
plt.text(0.5*maxtheta, np.mean(differences)+1.96*np.std(differences)+vertical_margin, "+1.96SD = %0.05f"
         % (np.mean(differences)+1.96*np.std(differences)))
plt.text(0.5*maxtheta, np.mean(differences)-1.96*np.std(differences)+vertical_margin, "-1.96SD = %0.05f"
         % (np.mean(differences)-1.96*np.std(differences)))
plt.show()

