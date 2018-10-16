import tensorflow as tf
from tensorflow import keras

def build_model():
	inputs = keras.Input(shape=(trX.shape[1],))

	x = keras.layers.Dense(64, activation='relu')(inputs)
	x = keras.layers.Dense(64, activation='relu')(x)
	predictions = keras.layers.Dense(10, activation='linear')(x)

	model = keras.Model(inputs=inputs, outputs=predictions)
	return model

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

model = build_model()
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['mse'])

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
