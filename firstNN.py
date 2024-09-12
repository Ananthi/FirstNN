import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from BoxOfficePredictionML import load_data, compute_cost
import matplotlib.pyplot as plt

X_train, Y_train, X_test, Y_test = load_data()
m, n = X_train.shape

model = Sequential(
    [
        tf.keras.Input(shape=(19,)),  # specify input size
        ### START CODE HERE ###
        Dense(400, activation="relu", name="layer1"),
        Dense(125, activation="relu", name="layer2"),
        Dense(70, activation="relu", name="layer3"),
        Dense(19, activation="relu", name="layer4"),
        Dense(1, activation="linear", name="layer5"),
        ### END CODE HERE ###
    ],
    name="my_model",
)
model.summary()
[layer1, layer2, layer3, layer4, layer5] = model.layers
# W1, b1 = layer1.get_weights()
# W2, b2 = layer2.get_weights()
W, b = layer5.get_weights()

# print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

# print(f"Layer 2 Weights: {W2}")

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.RMSprop(0.001),
)

model.fit(X_train, Y_train, epochs=20)
Y_p = []
for i in range(m):
    prediction = model.predict(X_train[i].reshape(1, n))
    Y_p.append(prediction[0, 0])
    print(
        f"Input no:{i}, Input:{X_train[i]},Prediction:{prediction},Actual:{Y_train[i]}"
    )

cost = compute_cost(X_train, Y_p, W, b)
print(f"Cost={cost}")
# layer.get_weights()
# a1 = layer(X_train[0].reshape(19, 1))
# w, b = layer.get_weights()
# # print(a1, w, b)
# prediction_tf = layer(X_train.reshape(-1, 1))
# print(X_train[1], prediction_tf)

plt.scatter(X_train[0], Y_train, c="r")

plt.scatter(X_train[0], Y_p)
# plt.show()
