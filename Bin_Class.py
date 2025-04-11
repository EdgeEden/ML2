import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.api.layers import Dense
from keras.api.losses import BinaryCrossentropy

matplotlib.use('TkAgg')
model = Sequential([
    Dense(units=30, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='sigmoid'),
])

X = np.array([17.99, 20.57, 19.69, 11.42, 20.29, 12.45, 18.25, 13.71, 13, 12.46,
              13.54, 13.08, 9.504, 13.03, 8.196, 12.05, 13.49, 11.76, 13.64, 11.94
              ]).reshape(-1, 1)
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0
              ])

model.compile(loss=BinaryCrossentropy())
history = model.fit(X, Y, epochs=1000)

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.show()

X_test = np.array([15.78, 13.56, 9.34, 14.56, 19.78]).reshape(-1, 1)
Y_test = model.predict(X_test)
print(Y_test)
