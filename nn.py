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

train_predictions = scalerY.inverse_transform(model.predict(trX).reshape(-1,1))
tst_preds = scalerY.inverse_transform(model.predict(teX))
plt.rcParams['figure.figsize'] = (15,3)
plt.plot(hist.history['mean_squared_error'], label='train', color='black')
plt.plot(hist.history['val_mean_squared_error'], label='test', color='blue')
plt.title('Train and test MSE of each epoch (during ANN fitting)')
plt.legend()
plt.show()

train_resid = pd.Series(
	(scalerY.inverse_transform(trY) - train_predictions).ravel(),
	name='Train residuals')
test_resid = pd.Series((scalerY.inverse_transform(teY) - tst_preds).ravel(),
					   name='Test residuals')

plt.plot(test_resid)
plt.show()
plt.hist(test_resid)
plt.show()

# interactive_plot(train_resid, title=train_resid.name)
# interactive_plot(test_resid, title=test_resid.name)

train_preds_df = pd.DataFrame({'Train prediction': train_predictions.ravel(),
							   'Train true values': scalerY.inverse_transform(
								   trY).ravel()},
							  index=trainset.index)

t = plotly_fname.split(".")[0]
interactive_plot(train_preds_df, title="Training result of %s" % t,
				 filename=t + '_train_preds.html')

test_preds_df = pd.DataFrame({'Test prediction': tst_preds.ravel(),
							  'Test true values': scalerY.inverse_transform(
								  teY).ravel()},
							 index=testset.index)
interactive_plot(test_preds_df, title=plt_title, filename=plotly_fname)

