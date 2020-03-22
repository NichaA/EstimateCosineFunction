# Estimate the cosine function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Assign number of available data points and resolution for X
N = 10000
theta_resolution = 0.01

# create the training data set
theta = np.arange(-180, 180+theta_resolution, theta_resolution)
X_train = np.random.choice(theta, N)
y_train = np.cos(X_train*np.pi/180) * (1 + np.random.random_integers(-1000, 1000, X_train.shape)/10000)

# make sure that the data set is correct and within the min/max envelopes
maxenvelope = np.cos(theta*np.pi/180) * 1.1
minenvelope = np.cos(theta*np.pi/180) * 0.9
plt.scatter(X_train, y_train, s=1)
plt.plot(theta, minenvelope, '-b', theta, maxenvelope, '-r')
plt.xlabel('X (degrees)')
plt.ylabel('Y')
plt.show()

# normalize X to [-1,1]
X_train /= 180.

# split the data into training and test sets (not in use, instead create X_test using theta)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_test = theta/180.
y_test = np.cos(X_test*np.pi)

# define the keras model
model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# compile the keras model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# fit the keras model on the data set
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=100, verbose=1, callbacks=[es])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim((0, 0.005))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# make predictions with the model
predictions = np.squeeze(model.predict(X_test))

# display the first 10 cases
for i in range(10):
    print('%s => %f (expected %f)' % (X_test[i].tolist(), predictions[i], y_test[i]))

# evaluate the model
testmse = mean_squared_error(y_test, predictions)
print('MSE: %.5f' % (testmse))
print('RMSE: %.5f' % np.sqrt(testmse))

# observe the results
plt.scatter(X_test*180, predictions)
plt.plot(X_test*180, np.cos(X_test*np.pi), '-r')
plt.show()

# observe the residuals with the Bland-Altman plot
differences = predictions - y_test

mintheta = np.min(theta)
maxtheta = np.max(theta)
plt.scatter(X_test*180, differences)
plt.hlines(np.mean(differences), mintheta, maxtheta)
plt.hlines(np.mean(differences)+1.96*np.std(differences), mintheta, maxtheta, linestyles='dashed')
plt.hlines(np.mean(differences)-1.96*np.std(differences), mintheta, maxtheta, linestyles='dashed')
plt.text(0.5*maxtheta, np.mean(differences)+0.005, "Mean = %0.05f" % np.mean(differences))
plt.text(0.5*maxtheta, np.mean(differences)+1.96*np.std(differences)+0.005, "+1.96SD = %0.05f" % (np.mean(differences)+1.96*np.std(differences)))
plt.text(0.5*maxtheta, np.mean(differences)-1.96*np.std(differences)+0.005, "-1.96SD = %0.05f" % (np.mean(differences)-1.96*np.std(differences)))
plt.show()

