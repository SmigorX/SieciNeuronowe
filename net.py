import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# Function to log parameters and results
def log_results_to_file(filename, params, results):
    with open(filename, "a") as file:
        file.write("Learning Parameters:\n")
        for key, value in params.items():
            file.write(f"{key}: {value}\n")
        file.write("\nResults:\n")
        for key, value in results.items():
            file.write(f"{key}: {value:.4f}\n")
        file.write("\n" + "=" * 50 + "\n")


# 1. Load data
data = pd.read_csv("./diabetes.csv")

# 2. Display first few rows
print("First few rows of the dataset:")
print(data.head())

# 3. Split into features and labels
X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]  # Labels

# 4. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Display shapes
print("\nShapes:")
print("X_train:", X_train_scaled.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test_scaled.shape)
print("y_test:", y_test.shape)

# 7. Define hyperparameter permutations
optimizer_learning_rate = [0.003]
fitting_batch = [10, 15, 20, 25, 30]
stop_patience = [20, 25]
reduce_learning_factor = [0.50, 0.51, 0.52, 0.53]
reduce_learning_patience = [10, 15, 20]
reduce_learning_minimal = 1e-6
fitting_epochs = 500

# 8. Create a list of all permutations
permutations = list(
    itertools.product(
        optimizer_learning_rate,
        stop_patience,
        reduce_learning_factor,
        reduce_learning_patience,
        fitting_batch,
    )
)

# 9. Initialize variables to store the best results and all results
best_test_accuracy = 0
best_params = {}
best_results = {}

all_results = []

# 10. Iterate through all permutations
for lr, patience, factor, reduce_patience, batch_size in permutations:
    print(
        f"\nTraining with lr={lr}, patience={patience}, factor={factor}, "
        f"reduce_patience={reduce_patience}, batch_size={batch_size}"
    )

    # 11. Define the model
    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu", input_shape=(X_train_scaled.shape[1],)),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 12. Compile the model
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # 13. Define callbacks
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        mode="max",
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=factor,
        patience=reduce_patience,
        min_lr=reduce_learning_minimal,
        mode="max",
        verbose=0,
    )

    # 14. Train the model
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=fitting_epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    # 15. Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    train_accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]

    # 16. Log parameters and results
    params = {
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": len(history.history["accuracy"]),
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "early_stopping_patience": patience,
        "reduce_lr_factor": factor,
        "reduce_lr_patience": reduce_patience,
    }

    results = {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
    }

    log_results_to_file("learning_results.log", params, results)

    # Store results for plotting
    all_results.append(
        {
            "learning_rate": lr,
            "patience": patience,
            "factor": factor,
            "reduce_patience": reduce_patience,
            "batch_size": batch_size,
            "test_accuracy": test_accuracy,
        }
    )

    # 17. Update best results
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_params = params
        best_results = results

    # 18. Print current results
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 19. Print the best results
print("\nBest Results:")
print("Parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")
print("\nResults:")
for key, value in best_results.items():
    print(f"{key}: {value:.4f}")

# 20. Plot training and validation accuracy for the best model
best_model = keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(1, activation="sigmoid"),
    ]
)

best_optimizer = Adam(learning_rate=best_params["learning_rate"])
best_model.compile(
    optimizer=best_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)

best_early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=best_params["early_stopping_patience"],
    mode="max",
    restore_best_weights=True,
)

best_reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=best_params["reduce_lr_factor"],
    patience=best_params["reduce_lr_patience"],
    min_lr=reduce_learning_minimal,
    mode="max",
    verbose=1,
)

best_history = best_model.fit(
    X_train_scaled,
    y_train,
    epochs=fitting_epochs,
    batch_size=best_params["batch_size"],
    validation_split=0.2,
    callbacks=[best_early_stopping, best_reduce_lr],
    verbose=1,
)

plt.plot(best_history.history["accuracy"], label="Training Accuracy")
plt.plot(best_history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Best Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 21. Plot parameter influence with averaged values
results_df = pd.DataFrame(all_results)

# Plot for each parameter
plt.figure(figsize=(15, 10))

# Learning Rate
plt.subplot(2, 3, 1)
avg_lr = results_df.groupby("learning_rate")["test_accuracy"].mean()
plt.plot(avg_lr.index, avg_lr.values, marker="o")
plt.title("Learning Rate vs Avg Test Accuracy")
plt.xlabel("Learning Rate")
plt.ylabel("Avg Test Accuracy")
plt.xticks(optimizer_learning_rate)
plt.grid(True)

# Batch Size
plt.subplot(2, 3, 2)
avg_bs = results_df.groupby("batch_size")["test_accuracy"].mean()
plt.plot(avg_bs.index, avg_bs.values, marker="o")
plt.title("Batch Size vs Avg Test Accuracy")
plt.xlabel("Batch Size")
plt.ylabel("Avg Test Accuracy")
plt.xticks(fitting_batch)
plt.grid(True)

# Early Stopping Patience
plt.subplot(2, 3, 3)
avg_patience = results_df.groupby("patience")["test_accuracy"].mean()
plt.plot(avg_patience.index, avg_patience.values, marker="o")
plt.title("Early Stopping Patience vs Avg Test Accuracy")
plt.xlabel("Early Stopping Patience")
plt.ylabel("Avg Test Accuracy")
plt.xticks(stop_patience)
plt.grid(True)

# Reduce LR Factor
plt.subplot(2, 3, 4)
avg_factor = results_df.groupby("factor")["test_accuracy"].mean()
plt.plot(avg_factor.index, avg_factor.values, marker="o")
plt.title("Reduce LR Factor vs Avg Test Accuracy")
plt.xlabel("Reduce LR Factor")
plt.ylabel("Avg Test Accuracy")
plt.xticks(reduce_learning_factor)
plt.grid(True)

# Reduce LR Patience
plt.subplot(2, 3, 5)
avg_reduce_patience = results_df.groupby("reduce_patience")["test_accuracy"].mean()
plt.plot(avg_reduce_patience.index, avg_reduce_patience.values, marker="o")
plt.title("Reduce LR Patience vs Avg Test Accuracy")
plt.xlabel("Reduce LR Patience")
plt.ylabel("Avg Test Accuracy")
plt.xticks(reduce_learning_patience)
plt.grid(True)

plt.tight_layout()
plt.show()
