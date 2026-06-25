import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from classification import build_widenet
from training_evaluation import train_model

# Load fused features
X_train = np.load('TN5000_features.npy')
y_train = np.load('TN5000_labels.npy')

X_test = np.load('DDTI_features.npy')
y_test = np.load('DDTI_labels.npy')

# Label encoding
encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build model
model = build_widenet(X_train.shape[1])

# Train model
train_model(
    model,
    X_train,
    y_train_cat,
    X_test,
    y_test_cat
)

# Prediction
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

# Evaluation
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)
