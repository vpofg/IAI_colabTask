import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import numpy as np
from IPython.display import Image, display
import pydotplus
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Precision, Recall, AUC

df = pd.read_csv('DATASETS/diabetes.csv')
print(df.head())

# Check the type of columns in df
print(df.dtypes)

# Count the number of instances, calculate means, variances, etc.
print(df.describe())

# Show how many classes are there
print(df['Outcome'].value_counts())

x = df.drop('Outcome', axis=1).values
y = df['Outcome'].to_numpy()

print("The type of x is:" + str(x.dtype))
print("The type of y is:" + str(y.dtype))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


### NEURAL NETWORK TRAINING ###

NB_EPOCHS = 1000
BATCH_SIZE = 16 

model = Sequential()

# Adding layers
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Adding new metrics Precision, Recall, AUC
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall(), AUC()])

print('Starting training...')

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=NB_EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1)

print('Training finished.')


instances = X_test[:5]
predictions = model.predict(instances)
predicted_classes = (predictions > 0.5).astype("int32")

print("Predicted probabilities for the first 5 instances:", predictions)
print("Predicted classes for the first 5 instances:", predicted_classes)


new_instances = np.array([
    [5, 116, 74, 0, 0, 25.6, 0.201, 30],
    [10, 150, 78, 32, 0, 35.4, 0.282, 45]
])
new_predictions = model.predict(new_instances)
new_predicted_classes = (new_predictions > 0.5).astype("int32")

print("Predicted probabilities for new instances:", new_predictions)
print("Predicted classes for new instances:", new_predicted_classes)


train_loss, train_accuracy, train_precision, train_recall, train_auc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {test_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Validation Precision: {test_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Validation Recall: {test_recall:.4f}")
print(f"Training AUC: {train_auc:.4f}")
print(f"Validation AUC: {test_auc:.4f}")

# Loss curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.savefig('loss_curves.png')

# Accuracy curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.savefig('accuracy_curves.png')

# Precision curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision Curves')
plt.savefig('precision_curves.png')

# Recall curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.title('Recall Curves')
plt.savefig('recall_curves.png')

# AUC curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.title('AUC Curves')
plt.savefig('auc_curves.png')   