import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEPENDENT_DATASET_PATH = os.path.join(BASE_DIR, "data", "independent_test_set")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_CSV = os.path.join(PLOTS_DIR, "evaluation_results.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_emnist_test_set():
    x_test_emnist = np.load(os.path.join(INDEPENDENT_DATASET_PATH, "x_test_emnist.npy"))
    y_test_emnist = np.load(os.path.join(INDEPENDENT_DATASET_PATH, "y_test_emnist.npy"))
    return x_test_emnist, y_test_emnist

def evaluate_models():
    x_test, y_test = load_emnist_test_set()
    x_test = x_test.reshape(-1, 28, 28, 1)  # Ensure correct shape
    results = []

    for model_name in os.listdir(MODEL_DIR):
        if model_name.endswith(".h5"):
            model_path = os.path.join(MODEL_DIR, model_name)
            print(f"Evaluating model: {model_name}")

            model = load_model(model_path)
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            print(f"Model: {model_name} - Loss: {loss}, Accuracy: {accuracy}")

            results.append({
                "Model Name": model_name,
                "Loss": loss,
                "Accuracy": accuracy
            })

            y_pred = model.predict(x_test).argmax(axis=1)
            cm = confusion_matrix(y_test, y_pred)

            # Save confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
            plt.title(f"Confusion Matrix for {model_name}")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            cm_plot_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name.replace('.h5', '')}.png")
            plt.savefig(cm_plot_path)
            plt.close()

            print(f"Confusion matrix saved to {cm_plot_path}")

    # Add this at the end of evaluate_models() in model_evaluator.py
    if results:
        best_model = max(results, key=lambda x: x['Accuracy'])
        best_model_name = best_model["Model Name"]
        best_model_path = os.path.join(MODEL_DIR, best_model_name)

        # Save the best model name to a file
        best_model_file = os.path.join(MODEL_DIR, "best_model.txt")
        with open(best_model_file, "w") as f:
            f.write(best_model_path)

        print(f"Best model saved to {best_model_file}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Evaluation results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    evaluate_models()
