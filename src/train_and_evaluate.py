import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
from sklearn.metrics import accuracy_score

# ==================================================
# DATA LOADING
# ==================================================

# Load the heart failure dataset
df = pd.read_csv("../data/heart.csv")

# ==================================================
# FEATURE ENGINEERING
# ==================================================

# Convert categorical variables into numerical
# representations using one-hot encoding.
df = pd.get_dummies(
    df,
    columns=[
        "Sex",
        "ST_Slope",
        "ExerciseAngina",
        "RestingECG",
        "ChestPainType"
    ]
)
# ==================================================
# FEATURE / TARGET SPLIT
# ==================================================

# Input features
X = df.drop("HeartDisease", axis=1)

# Target variable
y = df["HeartDisease"]

# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# FEATURE SCALING
# ==================================================

# Neural networks perform better on standardized
# numerical features.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================================
# MODEL ARCHITECTURE
# ==================================================

model = Sequential()

# Input layer
model.add(Input(shape=(X_train.shape[1],)))

# Hidden layer 1
model.add(Dense(
    units=6,
    activation="relu"
))

# Hidden layer 2
model.add(Dense(
    units=6,
    activation="relu"
))

# Output layer
model.add(Dense(
    units=1,
    activation="sigmoid"
))

# ==================================================
# MODEL COMPILATION
# ==================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# ==================================================
# MODEL TRAINING
# ==================================================
#
# Uncomment this section when training
# a new model from scratch.
#
# model.fit(
#     X_train,
#     y_train,
#     validation_split=0.33,
#     batch_size=10,
#     epochs=100
# )
#
# Persist trained model to disk.
# The saved model can later be loaded
# for inference and evaluation without retraining.
#
# model.save("model.h5")
#
# ==================================================

# ==================================================
# MODEL LOADING
# ==================================================

# Load serialized model from disk.
# The model was previously trained and saved
# using model.save("model.h5").
model = tf.keras.models.load_model("model.h5")

# ==================================================
# MODEL INFERENCE
# ==================================================

# Generate prediction probabilities
y_pred = model.predict(X_test)

# Convert probabilities into binary predictions.
# Values greater than 0.5 indicate predicted
# heart disease, while lower values indicate
# no heart disease.
y_pred = y_pred > 0.5

# ==================================================
# SAMPLE PREDICTIONS
# ==================================================

# Display first 10 predictions.
print("\nSample Predictions:")
print(y_pred.tolist()[:10])

# ==================================================
# MODEL EVALUATION
# ==================================================

# Calculate classification accuracy
# on the test dataset.
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nModel Accuracy: {accuracy:.4f}")