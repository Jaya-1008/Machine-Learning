import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# A1(a) ENCODING
def encode_data(X):
    return np.asarray(X, dtype=float)
def calculate_mean(values):
    valid_values = [value for value in values if not np.isnan(value)]
    if len(valid_values) == 0:
        return 0
    return sum(valid_values) / len(valid_values)
def calculate_median(values):
    valid_values = [value for value in values if not np.isnan(value)]
    if len(valid_values) == 0:
        return 0
    valid_values = sorted(valid_values)
    n = len(valid_values)
    if n % 2 == 1:
        return valid_values[n // 2]
    return (valid_values[n // 2 - 1] +valid_values[n // 2]) / 2
def calculate_mode(values):
    valid_values = [value for value in values if not np.isnan(value)]
    if len(valid_values) == 0:
        return 0
    frequency = {}
    for value in valid_values:
        frequency[value] = frequency.get(value, 0) + 1
    maximum_frequency = max(frequency.values())
    mode_values = [
        value for value in frequency
        if frequency[value] == maximum_frequency
    ]
    return min(mode_values)
def impute_missing_values(data, method="mean"):
    data = np.asarray(data, dtype=float).copy()
    for column in range(data.shape[1]):
        column_values = data[:, column]
        if np.isnan(column_values).any():
            if method == "mean":
                replacement = calculate_mean(column_values)
            elif method == "median":
                replacement = calculate_median(column_values)
            elif method == "mode":
                replacement = calculate_mode(column_values)
            else:
                raise ValueError("Method must be mean, median or mode.")
            for row in range(data.shape[0]):
                if np.isnan(data[row, column]):
                    data[row, column] = replacement
    return data
# A1(c) DISTANCE CALCULATION
def euclidean_distance(point1, point2):
    total = 0
    for i in range(len(point1)):
        total += (point1[i] - point2[i]) ** 2
    return np.sqrt(total)
# A1(d) SORTING ALGORITHMS
def bubble_sort(items):
    items = items.copy()
    n = len(items)
    for i in range(n):
        for j in range(0, n - i - 1):
            if items[j][0] > items[j + 1][0]:
                items[j], items[j + 1] = (items[j + 1],items[j])
    return items
def selection_sort(items):
    items = items.copy()
    n = len(items)
    for i in range(n):
        minimum_index = i
        for j in range(i + 1, n):
            if (items[j][0] < items[minimum_index][0] or (items[j][0] == items[minimum_index][0] and items[j][1] < items[minimum_index][1])):
                minimum_index = j
        items[i], items[minimum_index] = (items[minimum_index],items[i])
    return items
def insertion_sort(items):
    items = items.copy()
    for i in range(1, len(items)):
        current = items[i]
        j = i - 1
        while j >= 0:
            if (items[j][0] > current[0] or (items[j][0] == current[0] and items[j][1] > current[1])):
                items[j + 1] = items[j]
                j -= 1
            else:
                break
        items[j + 1] = current
    return items
def sort_distances(items, algorithm="selection"):
    if algorithm == "bubble":
        return bubble_sort(items)
    elif algorithm == "selection":
        return selection_sort(items)
    elif algorithm == "insertion":
        return insertion_sort(items)
    else:
        raise ValueError("Choose bubble, selection or insertion.")
# A1(e) IDENTIFY K NEAREST NEIGHBORS
def identify_neighbors(distances,training_labels,k,algorithm="selection"):
    items = []
    for index in range(len(distances)):
        items.append((distances[index],index,training_labels[index]))
    sorted_items = sort_distances(items, algorithm)
    return sorted_items[:k]
# A1(f) CLASS EVALUATION AND ASSIGNMENT
def majority_vote(neighbors):
    class_counts = {}
    for neighbor in neighbors:
        label = neighbor[2]
        class_counts[label] = (class_counts.get(label, 0) + 1)
    maximum_count = max(class_counts.values())
    candidate_classes = [label
        for label in class_counts
        if class_counts[label] == maximum_count
    ]
    for neighbor in neighbors:
        if neighbor[2] in candidate_classes:
            return neighbor[2]
    return candidate_classes[0]
# CUSTOM KNN CLASSIFIER
def custom_knn_predict(X_train,y_train,X_test,k=3,algorithm="selection"):
    predictions = []
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train - test_point) ** 2,axis=1))
        neighbors = identify_neighbors(distances,y_train,k,algorithm)
        predicted_class = majority_vote(neighbors)
        predictions.append(predicted_class)
    return np.array(predictions)
# A7 - FIT()
class CustomKNN:
    def __init__(self, k=3, algorithm="selection"):
        self.k = k
        self.algorithm = algorithm
        self.X_train = None
        self.y_train = None
    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
    # A7 - PREDICT()
    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        return custom_knn_predict(self.X_train,self.y_train,X,self.k,self.algorithm)
    # A7 - SCORE()
    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        correct = np.sum(predictions == y)
        accuracy = correct / len(y)
        return accuracy
    
# MAIN PROGRAM
if __name__ == "__main__":
    # LOAD DATASET
    print("DIGITS DATASET")
    digits = load_digits()
    X = digits.data
    y = digits.target
    print("\nOriginal dataset shape:", X.shape)
    print("Original number of classes:", len(np.unique(y)))
    # A3 - SELECT TWO CLASSES
    # Assignment says to use only two classes.
    # We select digits 0 and 1.
    selected_classes = [0, 1]
    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]
    print("\nClasses selected:", selected_classes)
    print("Dataset shape after selecting two classes:", X.shape)
    print("\nClass distribution:")
    for class_label in selected_classes:print("Digit",class_label,":",np.sum(y == class_label),"samples")
    # A1(a) ENCODING
    X = encode_data(X)
    print("\nA1(a): Encoding completed.")
    print("Data type:", X.dtype)
    # A1(b) DATA IMPUTATION
    X = impute_missing_values(X,method="mean")
    print("A1(b): Mean-based imputation completed.")
    print("Missing values:", np.isnan(X).sum())
    # A1(c) DISTANCE
    sample_distance = euclidean_distance(X[0],X[1])
    print("\nA1(c): Euclidean Distance")
    print("Distance between first two samples:",round(sample_distance, 4))
    # A1(d) SORTING ALGORITHMS
    sample_items = [(5.2, 0, 0),(2.1, 1, 1),(3.7, 2, 0),(2.1, 3, 1)]
    print("\nA1(d): Sorting Algorithms")
    print("Bubble Sort:",bubble_sort(sample_items))
    print("Selection Sort:",selection_sort(sample_items))
    print("Insertion Sort:",insertion_sort(sample_items))
    # A3 - TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
    print("A3: TRAIN-TEST SPLIT")
    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    # A1(e) AND A1(f) DEMONSTRATION
    print("A1: CUSTOM KNN MODULES")
    demonstration_distances = np.sqrt(np.sum((X_train - X_test[0]) ** 2,axis=1))
    demonstration_neighbors = identify_neighbors(demonstration_distances,y_train,k=3,algorithm="selection")
    demonstration_prediction = majority_vote(demonstration_neighbors)
    print("\nFirst test sample true class:",y_test[0])
    print("3 nearest neighbors:",[neighbor[2] for neighbor in demonstration_neighbors])
    print("Neighbor distances:",[round(neighbor[0], 4) for neighbor in demonstration_neighbors])
    print("Predicted class:",demonstration_prediction)
    # A4 - SCIKIT-LEARN KNN CLASSIFIER
    print("A4: SCIKIT-LEARN KNN")
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    start_time = time.perf_counter()
    sklearn_knn.fit(X_train,y_train)
    sklearn_train_time = (time.perf_counter() - start_time)
    print("\nK value:",sklearn_knn.n_neighbors)
    print("Training time:",round(sklearn_train_time, 6),"seconds")
    # A5 - TEST ACCURACY
    print("A5: KNN ACCURACY")
    sklearn_accuracy = sklearn_knn.score(X_test,y_test)
    print("\nScikit-learn KNN Accuracy:",round(sklearn_accuracy, 4))
    print("Accuracy percentage:",round(sklearn_accuracy * 100, 2),"%")
    # A6 - PREDICT()
    print("A6: PREDICTION")
    sklearn_predictions = sklearn_knn.predict(X_test)
    print("\nFirst 20 actual labels:")
    print(y_test[:20])
    print("\nFirst 20 predicted labels:")
    print(sklearn_predictions[:20])
    # A7 - DEVELOPED KNN
    print("A7: DEVELOPED KNN")
    custom_model = CustomKNN(k=3,algorithm="selection")
    start_time = time.perf_counter()
    custom_model.fit(X_train,y_train)
    custom_fit_time = (time.perf_counter() - start_time)
    start_time = time.perf_counter()
    custom_predictions = custom_model.predict(X_test)
    custom_predict_time = (time.perf_counter() - start_time)
    custom_accuracy = custom_model.score(X_test,y_test)
    print("\nCustom KNN Accuracy:",round(custom_accuracy, 4))
    print("Accuracy percentage:",round(custom_accuracy * 100, 2),"%")
    print("Custom KNN prediction time:",round(custom_predict_time, 6),"seconds")
    print("\nFirst 20 custom predictions:")
    print(custom_predictions[:20])
    # A8 - COMPARISON FOR DIFFERENT K VALUES
    print("A8: CUSTOM KNN VS SCIKIT-LEARN KNN")
    # Different values of k
    k_values = [1, 3, 5, 7, 9]
    custom_accuracies = []
    sklearn_accuracies = []
    custom_times = []
    sklearn_times = []
    # Use selection sort for the custom KNN.
    sorting_algorithm = "selection"
    print("\nSorting algorithm used:",sorting_algorithm)
    print("\nAccuracy Comparison")
    print("-" * 60)
    print("{:<8}{:<20}{:<20}".format("K","Custom KNN","Scikit-learn"))
    for k_value in k_values:
        # CUSTOM KNN
        custom_model_k = CustomKNN(k=k_value,algorithm=sorting_algorithm)
        custom_model_k.fit(X_train,y_train)
        start_time = time.perf_counter()
        custom_accuracy_k = custom_model_k.score(X_test,y_test)
        custom_time_k = (time.perf_counter() - start_time)
        # SCIKIT-LEARN KNN
        sklearn_model_k = KNeighborsClassifier(n_neighbors=k_value)
        start_time = time.perf_counter()
        sklearn_model_k.fit(X_train,y_train)
        sklearn_accuracy_k = sklearn_model_k.score(X_test,y_test)
        sklearn_time_k = (time.perf_counter() - start_time)
        # STORE RESULTS
        custom_accuracies.append(custom_accuracy_k)
        sklearn_accuracies.append(sklearn_accuracy_k)
        custom_times.append(custom_time_k)
        sklearn_times.append(sklearn_time_k)
        print("{:<8}{:<20}{:<20}".format(k_value,round(custom_accuracy_k, 4),round(sklearn_accuracy_k, 4)))
    # FINAL RESULTS TABLE
    print("FINAL RESULTS")
    print("\n{:<8}{:<18}{:<18}{:<18}{:<18}".format("K","Custom Accuracy","Sklearn Accuracy","Custom Time","Sklearn Time"))
    print("-" * 85)
    for i in range(len(k_values)):
        print("{:<8}{:<18}{:<18}{:<18}{:<18}".format(k_values[i],round(custom_accuracies[i], 4),round(sklearn_accuracies[i], 4),round(custom_times[i], 6),round(sklearn_times[i], 6)))
    # BEST K
    best_custom_index = np.argmax(custom_accuracies)
    best_sklearn_index = np.argmax(sklearn_accuracies)
    print("\nBest K for Custom KNN:",k_values[best_custom_index])
    print("Best Custom KNN Accuracy:",round(custom_accuracies[best_custom_index] * 100,2),"%")
    print("\nBest K for Scikit-learn KNN:",k_values[best_sklearn_index])
    print("Best Scikit-learn Accuracy:",round(sklearn_accuracies[best_sklearn_index] * 100,2),"%")
    # A8 - ACCURACY PLOT
    plt.figure(figsize=(8, 5))
    plt.plot(k_values,custom_accuracies,marker="o",label="Developed KNN")
    plt.plot(k_values,sklearn_accuracies,marker="s",label="Scikit-learn KNN")
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: Developed KNN vs Scikit-learn KNN")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # TIME COMPARISON PLOT
    plt.figure(figsize=(8, 5))
    plt.plot(k_values,custom_times,marker="o",label="Developed KNN")
    plt.plot(k_values,sklearn_times,marker="s",label="Scikit-learn KNN")
    plt.xlabel("Value of k")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # COMPARISON OF PREDICTIONS FOR k = 3
    print("PREDICTION COMPARISON FOR k = 3")
    prediction_comparison = (custom_predictions == sklearn_predictions)
    print("\nNumber of identical predictions:",np.sum(prediction_comparison))
    print("Total test samples:",len(y_test))
    print("Predictions identical:",np.all(prediction_comparison))