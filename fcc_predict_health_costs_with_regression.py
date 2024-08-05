# Import libraries. You may or may not use all of these.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
if not os.path.exists("insurance.csv"):
    url = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
    req = Request(
        url=url, 
        headers={"User-Agent": "Mozilla/5.0"}
    )

    webpage = urlopen(req)

    with open("insurance.csv","wb") as output:
        output.write(webpage.read())
        
dataset = pd.read_csv('insurance.csv')
dataset.tail()

#convert categorical data to numbers
vocab = {'male': 0, 'female': 1,
         'yes': 0, 'no': 1,
         'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
for category in ['sex', 'smoker', 'region']:
    dataset[category] = dataset[category].map(vocab)

#split dataset into training and testing data
column_indices = {name: i for i, name in enumerate(dataset.columns)}
n = len(dataset)
train_dataset, test_dataset = dataset[0:int(n*0.8)], dataset[int(n*0.8):]


#split expenses column from training and testing datasets
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

#Data normalisation
normalizer = layers.Normalization()
normalizer.adapt(np.array(train_dataset))

#initialise training model
model = keras.Sequential([
    normalizer,
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
    ])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
              loss='mean_absolute_error',
              metrics=['mean_absolute_error','mean_squared_error'])

model.fit(train_dataset, train_labels, epochs=200, verbose=0)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset, verbose=0).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
plt.show()
