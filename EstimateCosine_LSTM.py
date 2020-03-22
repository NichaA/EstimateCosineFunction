# Estimate the cosine function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

# define LSTM
lstm = Sequential()
lstm.add(LSTM(15, activation='tanh', return_sequences=True))
lstm.add(LSTM(10, activation='tanh'))
lstm.add(Dense(1, activation='tanh'))
lstm.compile(optimizer='adam', loss='mse')


# create data set
def build_sequence(n_steps, x_resolution, add_noise):

    X = np.arange(-180, 180 + x_resolution, x_resolution)
    if add_noise:
        y = np.cos(X * np.pi / 180) * (1 + np.random.random_integers(-1000, 1000, X.shape) / 10000)
    else:
        y = np.cos(X * np.pi / 180)

    a, b = [], []

    for i in range(len(X)):
        last_idx = i + n_steps
        if last_idx > len(X)-1:
            break
        sequence_x, sequence_y = X[i:last_idx+1], y[last_idx]
        a.append(sequence_x)
        b.append(sequence_y)

    return np.array(a), np.array(b)


# create dataset for lstm
X_train, y_train = build_sequence(n_steps=15, x_resolution=.01, add_noise=True)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
print("X.shape = {}".format(X_train.shape))
print("y.shape = {}".format(y_train.shape))

X_test, y_test = build_sequence(n_steps=15, x_resolution=.013, add_noise=False)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# compile the model
lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# fit the model on the data set
history = lstm.fit(X_train, y_train, validation_split=0.05, epochs=100, batch_size=100, callbacks=[es], shuffle=True)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.ylim((0, 0.005))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# make predictions with the model
predictions = np.squeeze(lstm.predict(X_test))

# evaluate the model
testmse = mean_squared_error(y_test, predictions)
print('MSE: %.5f' % (testmse))
print('RMSE: %.5f' % np.sqrt(testmse))

# observe the results
X_test = X_test[:, -1, :]
plt.scatter(X_test, predictions)
plt.plot(X_test, y_test, '-r')
plt.show()

# observe the residuals with the Bland-Altman plot
differences = predictions - y_test

mintheta = np.min(X_test)
maxtheta = np.max(X_test)
plt.scatter(X_test, differences)
plt.hlines(np.mean(differences), mintheta, maxtheta)
plt.hlines(np.mean(differences)+1.96*np.std(differences), mintheta, maxtheta, linestyles='dashed')
plt.hlines(np.mean(differences)-1.96*np.std(differences), mintheta, maxtheta, linestyles='dashed')
plt.text(0.5*maxtheta, np.mean(differences)+0.005, "Mean = %0.05f" % np.mean(differences))
plt.text(0.5*maxtheta, np.mean(differences)+1.96*np.std(differences)+0.005, "+1.96SD = %0.05f"
         % (np.mean(differences)+1.96*np.std(differences)))
plt.text(0.5*maxtheta, np.mean(differences)-1.96*np.std(differences)+0.005, "-1.96SD = %0.05f"
         % (np.mean(differences)-1.96*np.std(differences)))
plt.show()