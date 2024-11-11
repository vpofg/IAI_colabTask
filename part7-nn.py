from sklearn import tree
import pandas as pd
import numpy as np
from IPython.display import Image, display
import pydotplus
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('DATASETS/diabetes.csv')
print(df.head())

# check type of columns in df
print(df.dtypes)

# count the number of instances, calculate means, variances, etc.
print(df.describe())

# show how many classes are there
print(df['Outcome'].value_counts())

x = df.drop('Outcome', axis=1).values
y = df['Outcome'].to_numpy()

print("The type of x is:" + str(x.dtype))
print("The type of y is:" + str(y.dtype))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0) # what is going on here?

### NEURAL NETWORK TRAINING ###


NB_EPOCHS = 1000
BATCH_SIZE = 16

model = Sequential()

# adding layers
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid' ))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics

print('Starting training...')

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=NB_EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1)