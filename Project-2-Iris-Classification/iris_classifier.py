# ---------------------------------------------------------
# DecodeLabs Artificial Intelligence Internship
# Project 2: Iris Flower Classification Using KNN
# ---------------------------------------------------------

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from sklearn.model_selection import (
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

def load_and_explore_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset and display its basic information.

    Returns:
        A DataFrame containing the input features and
        a Series containing the target labels.
    """

    # Load the built-in Iris dataset
    iris = load_iris()

    # Convert the input features into a pandas DataFrame
    features = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )

    # Convert numerical targets into flower names
    target = pd.Series(
        data=iris.target,
        name="target"
    ).map({
        0: iris.target_names[0],
        1: iris.target_names[1],
        2: iris.target_names[2]
    })

    print("=" * 60)
    print("IRIS FLOWER DATASET")
    print("=" * 60)

    print("\nDataset shape:")
    print(features.shape)

    print("\nFeature names:")
    for feature_name in features.columns:
        print(f"- {feature_name}")

    print("\nFirst five records:")
    print(features.head())

    print("\nFlower class distribution:")
    print(target.value_counts())

    print("\nMissing values:")
    print(features.isnull().sum())

    return features, target

def split_dataset(
    features: pd.DataFrame,
    target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    80% of the data is used for training.
    20% of the data is used for testing.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target
    )

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test

def select_best_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_k: int = 20
) -> int:
    """
    Select the best K value using five-fold cross-validation.
    """

    results = []

    print("\n" + "=" * 60)
    print("K VALUE SELECTION")
    print("=" * 60)

    for k_value in range(1, max_k + 1):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k_value
                )
            )
        ])

        validation_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        results.append({
            "K": k_value,
            "Mean CV Accuracy": validation_scores.mean(),
            "Standard Deviation": validation_scores.std()
        })

    results_table = pd.DataFrame(results)

    best_result = results_table.sort_values(
        by=["Mean CV Accuracy", "K"],
        ascending=[False, True]
    ).iloc[0]

    best_k = int(best_result["K"])
    best_accuracy = float(best_result["Mean CV Accuracy"])

    print("\nCross-validation results:")
    print(
        results_table.to_string(
            index=False,
            formatters={
                "Mean CV Accuracy": "{:.4f}".format,
                "Standard Deviation": "{:.4f}".format
            }
        )
    )

    print(f"\nBest K value: {best_k}")
    print(
        f"Best mean validation accuracy: "
        f"{best_accuracy:.2%}"
    )

    save_k_selection_chart(
        results_table,
        best_k
    )

    return best_k
    pr


def save_k_selection_chart(
    results_table: pd.DataFrame,
    best_k: int
) -> None:
    """Save a graph showing validation accuracy for each K."""

    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    output_file = output_directory / "k_selection.png"

    plt.figure(figsize=(9, 5))

    plt.plot(
        results_table["K"],
        results_table["Mean CV Accuracy"],
        marker="o"
    )

    best_accuracy = results_table.loc[
        results_table["K"] == best_k,
        "Mean CV Accuracy"
    ].iloc[0]

    plt.scatter(
        best_k,
        best_accuracy,
        s=120,
        label=f"Best K = {best_k}"
    )

    plt.title("KNN Validation Accuracy for Different K Values")
    plt.xlabel("Number of Neighbours (K)")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.xticks(results_table["K"])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nK-selection chart saved: {output_file}")
    return best_k

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Standardize the training and testing features.

    The scaler learns the mean and standard deviation
    only from the training data.
    """

    scaler = StandardScaler()

    # Learn from training data and scale it
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same learned values to scale testing data
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    print("\nFirst original training record:")
    print(X_train.iloc[0].values)

    print("\nFirst scaled training record:")
    print(X_train_scaled[0])

    print("\nScaled training feature means:")
    print(X_train_scaled.mean(axis=0))

    print("\nScaled training feature standard deviations:")
    print(X_train_scaled.std(axis=0))

    return X_train_scaled, X_test_scaled, scaler

def train_knn_model(
    X_train_scaled,
    y_train: pd.Series,
    n_neighbors: int
) -> KNeighborsClassifier:
    """Create and train a KNN classifier."""

    if n_neighbors is None:
        raise ValueError(
            "n_neighbors cannot be None. "
            "Check that select_best_k() returns best_k."
        )

    if not isinstance(n_neighbors, int):
        raise TypeError(
            "n_neighbors must be an integer."
        )

    if n_neighbors < 1:
        raise ValueError(
            "n_neighbors must be at least 1."
        )

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    print("\n" + "=" * 60)
    print("KNN MODEL TRAINING")
    print("=" * 60)

    print("\nAlgorithm: K-Nearest Neighbors")
    print(f"Number of neighbours: {n_neighbors}")
    print(f"Training samples used: {len(y_train)}")
    print("Model trained successfully.")

    return model

def evaluate_model(
    model: KNeighborsClassifier,
    X_test_scaled,
    y_test: pd.Series
):
    """
    Test the trained model and display evaluation results.

    Returns:
        The predictions made for the testing dataset.
    """

    # Predict the species of unseen testing flowers
    predictions = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    # Create the confusion matrix
    class_names = list(model.classes_)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_names
    )

    confusion_table = pd.DataFrame(
        matrix,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names]
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nTesting samples: {len(y_test)}")
    print(f"Correct predictions: {(predictions == y_test).sum()}")
    print(f"Incorrect predictions: {(predictions != y_test).sum()}")

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nConfusion matrix:")
    print(confusion_table)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    print("\nActual and predicted results:")

    results = pd.DataFrame({
        "Actual Species": y_test.reset_index(drop=True),
        "Predicted Species": predictions
    })

    results["Correct"] = (
        results["Actual Species"]
        == results["Predicted Species"]
    )

    print(results)

    return predictions

def save_confusion_matrix_chart(
    y_test: pd.Series,
    predictions,
    class_names
) -> None:
    """
    Create and save a visual confusion matrix.

    Args:
        y_test: Actual flower species.
        predictions: Species predicted by the model.
        class_names: Names of the flower classes.
    """

    # Create the outputs directory if it does not exist
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    output_file = output_directory / "confusion_matrix.png"

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=class_names,
        display_labels=class_names,
        cmap="Blues",
        values_format="d"
    )

    display.ax_.set_title("KNN Iris Classification Confusion Matrix")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX VISUALIZATION")
    print("=" * 60)

    print(f"\nChart saved successfully: {output_file}")

def predict_new_flower(
    model: KNeighborsClassifier,
    scaler: StandardScaler,
    feature_names
) -> str:
    """
    Predict the species of a new flower using its measurements.

    Measurements:
        Sepal length
        Sepal width
        Petal length
        Petal width
    """

    # A completely new flower sample
    new_flower = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=feature_names
    )

    # Apply the same scaler used for training data
    new_flower_scaled = scaler.transform(new_flower)

    # Predict the flower species
    predicted_species = model.predict(new_flower_scaled)[0]

    # Get confidence-like class probabilities from KNN
    probabilities = model.predict_proba(new_flower_scaled)[0]

    print("\n" + "=" * 60)
    print("NEW FLOWER PREDICTION")
    print("=" * 60)

    print("\nFlower measurements:")
    print(f"Sepal length: {new_flower.iloc[0, 0]} cm")
    print(f"Sepal width:  {new_flower.iloc[0, 1]} cm")
    print(f"Petal length: {new_flower.iloc[0, 2]} cm")
    print(f"Petal width:  {new_flower.iloc[0, 3]} cm")

    print(f"\nPredicted species: {predicted_species}")

    print("\nClass probabilities:")

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):
        print(f"{class_name}: {probability:.2%}")

    return predicted_species



def main() -> None:
    """Run the Iris classification project."""

    # Load and explore the dataset
    features, target = load_and_explore_data()

    print("\nData loaded successfully.")
    print(f"Total samples: {len(features)}")
    print(f"Total features: {features.shape[1]}")
    print(f"Total classes: {target.nunique()}")

    # Split the dataset
    X_train, X_test, y_train, y_test = split_dataset(
        features,
        target
    )

    print("\nDataset split completed successfully.")

    # Select the best number of neighbours
    best_k = select_best_k(
        X_train,
        y_train,
        max_k=20
    )
    print(f"\nSelected K returned to main: {best_k}")

    # Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    print("\nFeature scaling completed successfully.")

    # Train the KNN model using the selected K
    model = train_knn_model(
        X_train_scaled,
        y_train,
        n_neighbors=best_k
    )

    # Evaluate the model
    predictions = evaluate_model(
        model,
        X_test_scaled,
        y_test
    )

    # Save the confusion matrix
    save_confusion_matrix_chart(
        y_test,
        predictions,
        model.classes_
    )

    # Predict the species of a new flower
    predict_new_flower(
        model,
        scaler,
        features.columns
    )


if __name__ == "__main__":
    main()




