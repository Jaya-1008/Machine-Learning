import numpy as np
import matplotlib.pyplot as plt
import time
import unittest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# A1(a) ENCODING
def encode_data(X):
    return np.asarray(X, dtype=float)

# --- IMPROVED (Claude suggestion) ---------------------------------------
# calculate_mean / calculate_median / calculate_mode each repeated the same
# "drop NaNs, and return 0 if nothing is left" logic. Pulling that into one
# helper removes the duplication without changing what any of the three
# functions compute or return.
def _filter_valid(values):
    return [value for value in values if not np.isnan(value)]

def calculate_mean(values):
    valid_values = _filter_valid(values)
    if len(valid_values) == 0:
        return 0
    return sum(valid_values) / len(valid_values)

def calculate_median(values):
    valid_values = _filter_valid(values)
    if len(valid_values) == 0:
        return 0
    valid_values = sorted(valid_values)
    n = len(valid_values)
    if n % 2 == 1:
        return valid_values[n // 2]
    return (valid_values[n // 2 - 1] + valid_values[n // 2]) / 2

def calculate_mode(values):
    valid_values = _filter_valid(values)
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
                items[j], items[j + 1] = (items[j + 1], items[j])
    return items

def selection_sort(items):
    items = items.copy()
    n = len(items)
    for i in range(n):
        minimum_index = i
        for j in range(i + 1, n):
            if (items[j][0] < items[minimum_index][0] or
                    (items[j][0] == items[minimum_index][0] and items[j][1] < items[minimum_index][1])):
                minimum_index = j
        items[i], items[minimum_index] = (items[minimum_index], items[i])
    return items

def insertion_sort(items):
    items = items.copy()
    for i in range(1, len(items)):
        current = items[i]
        j = i - 1
        while j >= 0:
            if (items[j][0] > current[0] or
                    (items[j][0] == current[0] and items[j][1] > current[1])):
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
def identify_neighbors(distances, training_labels, k, algorithm="selection"):
    # IMPROVED (Claude suggestion): replaced the manual index loop with
    # zip(), which builds the same list of (distance, index, label) tuples
    # more directly. No change in output.
    items = list(zip(distances, range(len(distances)), training_labels))
    sorted_items = sort_distances(items, algorithm)
    return sorted_items[:k]

# --- IMPROVED (Claude suggestion) ---------------------------------------
# majority_vote and weighted_vote were doing the exact same three steps
# (accumulate a score per class, find the max, break ties by nearest
# neighbor) and only differed in *how much* each neighbor contributes to
# its class's score (1 vote vs. a distance-based weight). Factoring that
# shared logic into _vote_by_weight keeps both public functions with their
# original name/signature/behaviour, but removes the duplicated code.
def _vote_by_weight(neighbors, weight_fn):
    class_scores = {}
    for neighbor in neighbors:
        distance, _, label = neighbor
        class_scores[label] = class_scores.get(label, 0) + weight_fn(distance)
    maximum_score = max(class_scores.values())
    candidate_classes = [
        label for label in class_scores
        if class_scores[label] == maximum_score
    ]
    for neighbor in neighbors:
        if neighbor[2] in candidate_classes:
            return neighbor[2]
    return candidate_classes[0]

# A1(f) CLASS EVALUATION AND ASSIGNMENT
def majority_vote(neighbors):
    return _vote_by_weight(neighbors, weight_fn=lambda distance: 1)

# A2 - WEIGHTED KNN CLASSIFICATION
def weighted_vote(neighbors):
    epsilon = 1e-10
    return _vote_by_weight(neighbors, weight_fn=lambda distance: 1 / (distance + epsilon))

# --- IMPROVED (Claude suggestion) ---------------------------------------
# custom_knn_predict and weighted_knn_predict were identical apart from the
# voting function they called (majority_vote vs weighted_vote). Extracting
# the shared "for each test point: compute distances, find neighbors, vote"
# loop into _knn_predict_core means that loop only exists once. Both
# original functions are kept (same name, same arguments, same return
# value) as thin wrappers, so nothing about how you call them changes.
def _knn_predict_core(X_train, y_train, X_test, k, algorithm, vote_fn):
    predictions = []
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        neighbors = identify_neighbors(distances, y_train, k, algorithm)
        predictions.append(vote_fn(neighbors))
    return np.array(predictions)

def custom_knn_predict(X_train, y_train, X_test, k=3, algorithm="selection"):
    return _knn_predict_core(X_train, y_train, X_test, k, algorithm, majority_vote)

def weighted_knn_predict(X_train, y_train, X_test, k=3, algorithm="selection"):
    return _knn_predict_core(X_train, y_train, X_test, k, algorithm, weighted_vote)

# --- IMPROVED (Claude suggestion) ---------------------------------------
# CustomKNN and WeightedKNN had identical __init__, fit and score methods,
# and only differed in which predict function they called. Moving the
# shared code into a _BaseKNN class removes that duplication while keeping
# CustomKNN and WeightedKNN as separate, independently usable classes with
# exactly the same constructor and methods as before.
class _BaseKNN:
    def __init__(self, k=3, algorithm="selection"):
        self.k = k
        self.algorithm = algorithm
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        correct = np.sum(predictions == y)
        return correct / len(y)

# A2 - WEIGHTED KNN CLASSIFIER CLASS
class WeightedKNN(_BaseKNN):
    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        return weighted_knn_predict(self.X_train, self.y_train, X, self.k, self.algorithm)

# A7 - FIT() / PREDICT() / SCORE()
class CustomKNN(_BaseKNN):
    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        return custom_knn_predict(self.X_train, self.y_train, X, self.k, self.algorithm)


# ============================================================================
# A10 - AI-ASSISTED OPTIMIZED KNN (for performance comparison, Q10)
# ============================================================================
# This is NOT a rewrite of the algorithm -- it is still plain k-NN with a
# majority vote. The classification logic is identical to CustomKNN. Only
# three low-level implementation details are changed, and each is explained
# below. The goal is to remove Python-level loops (slow) and replace them
# with NumPy's vectorized/C-level operations (fast), which matters most as
# the number of training samples and test samples grows.

def _pairwise_distances(X_train, X_test):
    # OPTIMIZATION 1: compute Euclidean distances between EVERY test point and
    # EVERY training point in a single matrix operation, instead of looping
    # over test points one at a time in Python (as custom_knn_predict does).
    #
    # Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a . b)
    # so the whole (n_test x n_train) distance matrix can be built from a
    # couple of matrix multiplications, which NumPy runs in compiled code.
    train_sq = np.sum(X_train ** 2, axis=1)[np.newaxis, :]   # shape (1, n_train)
    test_sq = np.sum(X_test ** 2, axis=1)[:, np.newaxis]     # shape (n_test, 1)
    cross_term = X_test @ X_train.T                          # shape (n_test, n_train)
    squared_distances = test_sq + train_sq - 2 * cross_term
    # Tiny negative values can appear here from floating-point rounding
    # (a point's true distance to itself should be 0, not -1e-15) -- clip
    # them so the sqrt below never sees a negative number.
    squared_distances = np.maximum(squared_distances, 0)
    return np.sqrt(squared_distances)

def _majority_vote_fast(nearest_labels_row):
    # OPTIMIZATION 3: vectorized vote-counting with np.bincount instead of the
    # dict-based loop used by majority_vote(). This is fast, but it only
    # works for non-negative integer class labels (0, 1, 2, ...), which is
    # true for this dataset (digit labels), so it's a reasonable trade-off
    # here but less general than the original majority_vote().
    # NOTE on tie-breaking: majority_vote() breaks ties by picking the class
    # of the single nearest neighbor. np.argmax breaks ties by picking the
    # smallest class label instead. For k=3 (odd) on a 2-class problem this
    # difference essentially never triggers, but it is a real, documented
    # behavioural difference between the two implementations.
    counts = np.bincount(nearest_labels_row)
    return np.argmax(counts)

class OptimizedKNN(_BaseKNN):
    """AI-assisted, vectorized version of CustomKNN. Same k-NN + majority-vote
    algorithm; only the internal implementation is optimized (see the three
    OPTIMIZATION comments in this section for exactly what changed and why)."""

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction.")
        X_test = np.asarray(X, dtype=float)
        distances = _pairwise_distances(self.X_train, X_test)  # (n_test, n_train)

        k = min(self.k, distances.shape[1])

        # OPTIMIZATION 2: find the k nearest neighbors with np.argpartition,
        # which only needs to guarantee the k smallest values end up in the
        # first k positions (average O(n) per row). This replaces the
        # bubble/selection/insertion sorts used by sort_distances(), which
        # fully sort the *entire* training set (O(n^2)) just to read off the
        # first k entries. np.argpartition also runs over every test point's
        # row at once (axis=1), instead of Python looping row by row.
        partial_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        partial_dist = np.take_along_axis(distances, partial_idx, axis=1)
        # The k selected neighbors aren't in distance order yet, so sort just
        # those k values per row (cheap, since k is small).
        order = np.argsort(partial_dist, axis=1)
        nearest_idx = np.take_along_axis(partial_idx, order, axis=1)
        nearest_labels = self.y_train[nearest_idx]  # shape (n_test, k)

        predictions = np.array([_majority_vote_fast(row) for row in nearest_labels])
        return predictions


# ============================================================================
# UNIT TESTS
# ============================================================================
# Run just these tests from the command line with:
#     python -m unittest knn_assignment_improved -v
# (When the file is run directly with "python knn_assignment_improved.py",
# __name__ == "__main__" so these tests are NOT executed automatically --
# only the assignment's main program below runs, exactly like before.
# unittest imports this file as a module instead, so this block is skipped
# and only the TestCase classes get discovered and run.)

class TestEncodeData(unittest.TestCase):
    # NORMAL CASE: a plain list of lists should become a float ndarray
    def test_converts_list_to_float_array(self):
        result = encode_data([[1, 2], [3, 4]])
        self.assertEqual(result.dtype, np.dtype(float))
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    # EDGE CASE: empty input should still return an empty float array, not crash
    def test_empty_input(self):
        result = encode_data([])
        self.assertEqual(result.size, 0)

    # INVALID INPUT: non-numeric values cannot be converted to float
    def test_non_numeric_raises_value_error(self):
        with self.assertRaises(ValueError):
            encode_data([["a", "b"], ["c", "d"]])


class TestCalculateMean(unittest.TestCase):
    # NORMAL CASE: plain numbers, no missing values
    def test_normal_values(self):
        self.assertAlmostEqual(calculate_mean([1, 2, 3, 4]), 2.5)

    # NORMAL CASE: NaNs should be ignored, not counted as 0
    def test_ignores_nan(self):
        self.assertAlmostEqual(calculate_mean([1, np.nan, 3]), 2.0)

    # EDGE CASE: all values missing -> function is defined to return 0
    def test_all_nan_returns_zero(self):
        self.assertEqual(calculate_mean([np.nan, np.nan]), 0)

    # EDGE CASE: single value -> mean is just that value
    def test_single_value(self):
        self.assertEqual(calculate_mean([7]), 7)


class TestCalculateMedian(unittest.TestCase):
    # NORMAL CASE: odd number of values -> middle element
    def test_odd_count(self):
        self.assertEqual(calculate_median([3, 1, 2]), 2)

    # NORMAL CASE: even number of values -> average of two middle elements
    def test_even_count(self):
        self.assertEqual(calculate_median([1, 2, 3, 4]), 2.5)

    # NORMAL CASE: NaNs should be dropped before finding the median
    def test_ignores_nan(self):
        self.assertEqual(calculate_median([1, np.nan, 2, 3]), 2)

    # EDGE CASE: all values missing -> defined to return 0
    def test_all_nan_returns_zero(self):
        self.assertEqual(calculate_median([np.nan, np.nan]), 0)


class TestCalculateMode(unittest.TestCase):
    # NORMAL CASE: one clear most-frequent value
    def test_normal_mode(self):
        self.assertEqual(calculate_mode([1, 2, 2, 3]), 2)

    # EDGE CASE: tie between two modes -> function returns the smaller one
    def test_tie_returns_smaller_value(self):
        self.assertEqual(calculate_mode([1, 1, 2, 2]), 1)

    # EDGE CASE: all values missing -> defined to return 0
    def test_all_nan_returns_zero(self):
        self.assertEqual(calculate_mode([np.nan, np.nan]), 0)


class TestImputeMissingValues(unittest.TestCase):
    # NORMAL CASE: missing value in a column gets replaced by that column's mean
    def test_mean_imputation(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = impute_missing_values(data, method="mean")
        self.assertFalse(np.isnan(result).any())
        self.assertAlmostEqual(result[0, 1], 4.0)  # only value in that column

    # NORMAL CASE: median imputation on a column with more than one value
    def test_median_imputation(self):
        data = np.array([[1.0], [np.nan], [3.0]])
        result = impute_missing_values(data, method="median")
        self.assertAlmostEqual(result[1, 0], 2.0)

    # NORMAL CASE: mode imputation picks the most frequent existing value
    def test_mode_imputation(self):
        data = np.array([[1.0], [1.0], [np.nan]])
        result = impute_missing_values(data, method="mode")
        self.assertAlmostEqual(result[2, 0], 1.0)

    # EDGE CASE: no missing values at all -> data should come back unchanged
    def test_no_missing_values_unchanged(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = impute_missing_values(data, method="mean")
        np.testing.assert_array_equal(result, data)

    # INVALID INPUT: unsupported method string should raise ValueError
    def test_invalid_method_raises_value_error(self):
        data = np.array([[1.0, np.nan]])
        with self.assertRaises(ValueError):
            impute_missing_values(data, method="bogus")


class TestEuclideanDistance(unittest.TestCase):
    # NORMAL CASE: simple 3-4-5 triangle distance
    def test_normal_distance(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)

    # EDGE CASE: distance from a point to itself is 0
    def test_same_point_zero_distance(self):
        self.assertEqual(euclidean_distance([1, 2, 3], [1, 2, 3]), 0)

    # NORMAL CASE: works correctly with negative coordinates
    def test_negative_coordinates(self):
        self.assertAlmostEqual(euclidean_distance([-1, -1], [2, 3]), 5.0)


class TestSortingAlgorithms(unittest.TestCase):
    # Each item is (distance, index, label); sorting is by distance,
    # with index as the tie-breaker for equal distances.
    unsorted_items = [(5.2, 0, 0), (2.1, 1, 1), (3.7, 2, 0), (2.1, 3, 1)]
    expected = [(2.1, 1, 1), (2.1, 3, 1), (3.7, 2, 0), (5.2, 0, 0)]

    # NORMAL CASE: all three sorting algorithms should agree on the same result
    def test_bubble_sort(self):
        self.assertEqual(bubble_sort(self.unsorted_items), self.expected)

    def test_selection_sort(self):
        self.assertEqual(selection_sort(self.unsorted_items), self.expected)

    def test_insertion_sort(self):
        self.assertEqual(insertion_sort(self.unsorted_items), self.expected)

    # EDGE CASE: already-sorted input should stay unchanged
    def test_already_sorted(self):
        sorted_items = [(1.0, 0, 0), (2.0, 1, 1)]
        self.assertEqual(bubble_sort(sorted_items), sorted_items)

    # EDGE CASE: empty list should not raise an error
    def test_empty_list(self):
        self.assertEqual(bubble_sort([]), [])
        self.assertEqual(selection_sort([]), [])
        self.assertEqual(insertion_sort([]), [])

    # EDGE CASE: single-element list is trivially "sorted"
    def test_single_element(self):
        single = [(1.0, 0, 0)]
        self.assertEqual(insertion_sort(single), single)

    # NORMAL CASE: sort_distances should dispatch to the right algorithm
    def test_sort_distances_dispatch(self):
        self.assertEqual(sort_distances(self.unsorted_items, "bubble"), self.expected)
        self.assertEqual(sort_distances(self.unsorted_items, "selection"), self.expected)
        self.assertEqual(sort_distances(self.unsorted_items, "insertion"), self.expected)

    # INVALID INPUT: unknown algorithm name should raise ValueError
    def test_sort_distances_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            sort_distances(self.unsorted_items, "quicksort")


class TestIdentifyNeighbors(unittest.TestCase):
    # NORMAL CASE: should return the k closest (distance, index, label) tuples, sorted
    def test_returns_k_nearest_sorted(self):
        distances = [5.0, 1.0, 3.0, 2.0]
        labels = ["a", "b", "c", "d"]
        result = identify_neighbors(distances, labels, k=2)
        self.assertEqual(result, [(1.0, 1, "b"), (2.0, 3, "d")])

    # EDGE CASE: k=0 should return an empty list of neighbors
    def test_k_zero_returns_empty(self):
        result = identify_neighbors([1.0, 2.0], ["a", "b"], k=0)
        self.assertEqual(result, [])

    # EDGE CASE: k larger than the number of points just returns all of them
    def test_k_larger_than_data_returns_all(self):
        result = identify_neighbors([2.0, 1.0], ["a", "b"], k=5)
        self.assertEqual(len(result), 2)


class TestVoting(unittest.TestCase):
    # NORMAL CASE: majority_vote picks the label that appears most often
    def test_majority_vote_clear_winner(self):
        neighbors = [(1.0, 0, "cat"), (2.0, 1, "cat"), (3.0, 2, "dog")]
        self.assertEqual(majority_vote(neighbors), "cat")

    # EDGE CASE: tie in counts -> majority_vote returns the class of the nearest neighbor
    def test_majority_vote_tie_breaks_by_distance(self):
        neighbors = [(1.0, 0, "dog"), (2.0, 1, "cat")]
        self.assertEqual(majority_vote(neighbors), "dog")  # dog is closer (distance 1.0)

    # NORMAL CASE: weighted_vote gives closer neighbors more influence, so a
    # single very close neighbor can outweigh two farther ones
    def test_weighted_vote_close_neighbor_outweighs_far_ones(self):
        neighbors = [(0.1, 0, "cat"), (10.0, 1, "dog"), (10.0, 2, "dog")]
        self.assertEqual(weighted_vote(neighbors), "cat")

    # EDGE CASE: a neighbor at distance 0 should not raise a division-by-zero error
    # (the small epsilon in weighted_vote protects against this)
    def test_weighted_vote_zero_distance_no_crash(self):
        neighbors = [(0.0, 0, "cat"), (5.0, 1, "dog")]
        result = weighted_vote(neighbors)
        self.assertEqual(result, "cat")


class TestKnnPredictFunctions(unittest.TestCase):
    # A simple, clearly-separated dataset: small x-values are class 0,
    # large x-values are class 1.
    X_train = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_train = np.array([0, 0, 1, 1])

    # NORMAL CASE: a point near the "0" cluster should be predicted as class 0
    def test_custom_knn_predict_normal(self):
        X_test = np.array([[0.5]])
        result = custom_knn_predict(self.X_train, self.y_train, X_test, k=1)
        self.assertEqual(result[0], 0)

    # NORMAL CASE: same idea but with the weighted version, near the "1" cluster
    def test_weighted_knn_predict_normal(self):
        X_test = np.array([[10.5]])
        result = weighted_knn_predict(self.X_train, self.y_train, X_test, k=3)
        self.assertEqual(result[0], 1)

    # EDGE CASE: multiple test points at once should each get a prediction
    def test_predict_multiple_points(self):
        X_test = np.array([[0.2], [10.2]])
        result = custom_knn_predict(self.X_train, self.y_train, X_test, k=1)
        np.testing.assert_array_equal(result, np.array([0, 1]))


class TestCustomKNN(unittest.TestCase):
    X_train = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_train = np.array([0, 0, 1, 1])

    # NORMAL CASE: fit then predict should classify a clearly-separated point correctly
    def test_fit_predict(self):
        model = CustomKNN(k=1)
        model.fit(self.X_train, self.y_train)
        prediction = model.predict(np.array([[0.5]]))
        self.assertEqual(prediction[0], 0)

    # NORMAL CASE: score should report 100% accuracy on the (easy) training data itself
    def test_score_on_training_data(self):
        model = CustomKNN(k=1)
        model.fit(self.X_train, self.y_train)
        accuracy = model.score(self.X_train, self.y_train)
        self.assertEqual(accuracy, 1.0)

    # INVALID USE: predicting before fit() should raise a clear error, not crash silently
    def test_predict_before_fit_raises_error(self):
        model = CustomKNN(k=1)
        with self.assertRaises(ValueError):
            model.predict(np.array([[0.0]]))


class TestWeightedKNN(unittest.TestCase):
    X_train = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_train = np.array([0, 0, 1, 1])

    # NORMAL CASE: fit then predict should classify a clearly-separated point correctly
    def test_fit_predict(self):
        model = WeightedKNN(k=3)
        model.fit(self.X_train, self.y_train)
        prediction = model.predict(np.array([[10.5]]))
        self.assertEqual(prediction[0], 1)

    # NORMAL CASE: score should report 100% accuracy on the (easy) training data itself
    def test_score_on_training_data(self):
        model = WeightedKNN(k=1)
        model.fit(self.X_train, self.y_train)
        accuracy = model.score(self.X_train, self.y_train)
        self.assertEqual(accuracy, 1.0)

    # INVALID USE: predicting before fit() should raise a clear error
    def test_predict_before_fit_raises_error(self):
        model = WeightedKNN(k=1)
        with self.assertRaises(ValueError):
            model.predict(np.array([[0.0]]))

    # INVALID INPUT: an unsupported sorting algorithm name should raise ValueError
    # only once prediction is actually attempted (fit itself doesn't sort anything)
    def test_invalid_algorithm_raises_on_predict(self):
        model = WeightedKNN(k=1, algorithm="not_a_real_algorithm")
        model.fit(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            model.predict(np.array([[0.0]]))


class TestOptimizedKNN(unittest.TestCase):
    X_train = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_train = np.array([0, 0, 1, 1])

    # NORMAL CASE: should classify a clearly-separated point correctly, same
    # as CustomKNN would, since the underlying algorithm is unchanged
    def test_fit_predict_matches_custom_knn(self):
        optimized_model = OptimizedKNN(k=1)
        optimized_model.fit(self.X_train, self.y_train)
        custom_model = CustomKNN(k=1)
        custom_model.fit(self.X_train, self.y_train)
        X_test = np.array([[0.5], [10.5]])
        np.testing.assert_array_equal(
            optimized_model.predict(X_test), custom_model.predict(X_test)
        )

    # EDGE CASE: k larger than the number of training points shouldn't crash
    # (np.argpartition needs k to be capped at the number of available points)
    def test_k_larger_than_training_set(self):
        model = OptimizedKNN(k=100)
        model.fit(self.X_train, self.y_train)
        result = model.predict(np.array([[0.5]]))
        self.assertEqual(len(result), 1)

    # INVALID USE: predicting before fit() should raise a clear error
    def test_predict_before_fit_raises_error(self):
        model = OptimizedKNN(k=1)
        with self.assertRaises(ValueError):
            model.predict(np.array([[0.0]]))


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
    selected_classes = [0, 1]
    mask = np.isin(y, selected_classes)
    X = X[mask]
    y = y[mask]
    print("\nClasses selected:", selected_classes)
    print("Dataset shape after selecting two classes:", X.shape)
    print("\nClass distribution:")
    for class_label in selected_classes:
        print("Digit", class_label, ":", np.sum(y == class_label), "samples")

    # A1(a) ENCODING
    X = encode_data(X)
    print("\nA1(a): Encoding completed.")
    print("Data type:", X.dtype)

    # A1(b) DATA IMPUTATION
    X = impute_missing_values(X, method="mean")
    print("A1(b): Mean-based imputation completed.")
    print("Missing values:", np.isnan(X).sum())

    # A1(c) DISTANCE
    sample_distance = euclidean_distance(X[0], X[1])
    print("\nA1(c): Euclidean Distance")
    print("Distance between first two samples:", round(sample_distance, 4))

    # A1(d) SORTING ALGORITHMS
    sample_items = [(5.2, 0, 0), (2.1, 1, 1), (3.7, 2, 0), (2.1, 3, 1)]
    print("\nA1(d): Sorting Algorithms")
    print("Bubble Sort:", bubble_sort(sample_items))
    print("Selection Sort:", selection_sort(sample_items))
    print("Insertion Sort:", insertion_sort(sample_items))

    # A3 - TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print("A3: TRAIN-TEST SPLIT")
    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # A1(e) AND A1(f) DEMONSTRATION
    print("A1: CUSTOM KNN MODULES")
    demonstration_distances = np.sqrt(np.sum((X_train - X_test[0]) ** 2, axis=1))
    demonstration_neighbors = identify_neighbors(demonstration_distances, y_train, k=3, algorithm="selection")
    demonstration_prediction = majority_vote(demonstration_neighbors)
    print("\nFirst test sample true class:", y_test[0])
    print("3 nearest neighbors:", [neighbor[2] for neighbor in demonstration_neighbors])
    print("Neighbor distances:", [round(neighbor[0], 4) for neighbor in demonstration_neighbors])
    print("Predicted class:", demonstration_prediction)

    # A4 - SCIKIT-LEARN KNN CLASSIFIER
    print("A4: SCIKIT-LEARN KNN")
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    start_time = time.perf_counter()
    sklearn_knn.fit(X_train, y_train)
    sklearn_train_time = time.perf_counter() - start_time
    print("\nK value:", sklearn_knn.n_neighbors)
    print("Training time:", round(sklearn_train_time, 6), "seconds")

    # A5 - TEST ACCURACY
    print("A5: KNN ACCURACY")
    sklearn_accuracy = sklearn_knn.score(X_test, y_test)
    print("\nScikit-learn KNN Accuracy:", round(sklearn_accuracy, 4))
    print("Accuracy percentage:", round(sklearn_accuracy * 100, 2), "%")

    # A6 - PREDICT()
    print("A6: PREDICTION")
    sklearn_predictions = sklearn_knn.predict(X_test)
    print("\nFirst 20 actual labels:")
    print(y_test[:20])
    print("\nFirst 20 predicted labels:")
    print(sklearn_predictions[:20])

    # A7 - DEVELOPED KNN
    print("A7: DEVELOPED KNN")
    custom_model = CustomKNN(k=3, algorithm="selection")
    start_time = time.perf_counter()
    custom_model.fit(X_train, y_train)
    custom_fit_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    custom_predictions = custom_model.predict(X_test)
    custom_predict_time = time.perf_counter() - start_time
    custom_accuracy = custom_model.score(X_test, y_test)
    print("\nCustom KNN Accuracy:", round(custom_accuracy, 4))
    print("Accuracy percentage:", round(custom_accuracy * 100, 2), "%")
    print("Custom KNN prediction time:", round(custom_predict_time, 6), "seconds")
    print("\nFirst 20 custom predictions:")
    print(custom_predictions[:20])

    # ========================================================
    # A2 - WEIGHTED KNN
    # ========================================================
    print("\n" + "=" * 60)
    print("A2: WEIGHTED KNN")
    print("=" * 60)

    weighted_model = WeightedKNN(k=3, algorithm="selection")
    start_time = time.perf_counter()
    weighted_model.fit(X_train, y_train)
    weighted_predictions = weighted_model.predict(X_test)
    weighted_time = time.perf_counter() - start_time
    weighted_accuracy = weighted_model.score(X_test, y_test)

    print("\nK value:", weighted_model.k)
    print("Weighted KNN Accuracy:", round(weighted_accuracy, 4))
    print("Accuracy percentage:", round(weighted_accuracy * 100, 2), "%")
    print("Weighted KNN prediction time:", round(weighted_time, 6), "seconds")
    print("\nFirst 20 weighted KNN predictions:")
    print(weighted_predictions[:20])

    # A8 - COMPARISON FOR DIFFERENT K VALUES
    print("A8: CUSTOM KNN VS SCIKIT-LEARN KNN")
    k_values = [1, 3, 5, 7, 9]
    custom_accuracies = []
    sklearn_accuracies = []
    custom_times = []
    sklearn_times = []
    sorting_algorithm = "selection"
    print("\nSorting algorithm used:", sorting_algorithm)
    print("\nAccuracy Comparison")
    print("-" * 60)
    print("{:<8}{:<20}{:<20}".format("K", "Custom KNN", "Scikit-learn"))
    for k_value in k_values:
        # CUSTOM KNN
        custom_model_k = CustomKNN(k=k_value, algorithm=sorting_algorithm)
        custom_model_k.fit(X_train, y_train)
        start_time = time.perf_counter()
        custom_accuracy_k = custom_model_k.score(X_test, y_test)
        custom_time_k = time.perf_counter() - start_time
        # SCIKIT-LEARN KNN
        sklearn_model_k = KNeighborsClassifier(n_neighbors=k_value)
        start_time = time.perf_counter()
        sklearn_model_k.fit(X_train, y_train)
        sklearn_accuracy_k = sklearn_model_k.score(X_test, y_test)
        sklearn_time_k = time.perf_counter() - start_time
        # STORE RESULTS
        custom_accuracies.append(custom_accuracy_k)
        sklearn_accuracies.append(sklearn_accuracy_k)
        custom_times.append(custom_time_k)
        sklearn_times.append(sklearn_time_k)
        print("{:<8}{:<20}{:<20}".format(k_value, round(custom_accuracy_k, 4), round(sklearn_accuracy_k, 4)))

    # FINAL RESULTS TABLE
    print("FINAL RESULTS")
    print("\n{:<8}{:<18}{:<18}{:<18}{:<18}".format(
        "K", "Custom Accuracy", "Sklearn Accuracy", "Custom Time", "Sklearn Time"))
    print("-" * 85)
    for i in range(len(k_values)):
        print("{:<8}{:<18}{:<18}{:<18}{:<18}".format(
            k_values[i], round(custom_accuracies[i], 4), round(sklearn_accuracies[i], 4),
            round(custom_times[i], 6), round(sklearn_times[i], 6)))

    # BEST K
    best_custom_index = np.argmax(custom_accuracies)
    best_sklearn_index = np.argmax(sklearn_accuracies)
    print("\nBest K for Custom KNN:", k_values[best_custom_index])
    print("Best Custom KNN Accuracy:", round(custom_accuracies[best_custom_index] * 100, 2), "%")
    print("\nBest K for Scikit-learn KNN:", k_values[best_sklearn_index])
    print("Best Scikit-learn Accuracy:", round(sklearn_accuracies[best_sklearn_index] * 100, 2), "%")

    # A8 - ACCURACY PLOT
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, custom_accuracies, marker="o", label="Developed KNN")
    plt.plot(k_values, sklearn_accuracies, marker="s", label="Scikit-learn KNN")
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
    plt.plot(k_values, custom_times, marker="o", label="Developed KNN")
    plt.plot(k_values, sklearn_times, marker="s", label="Scikit-learn KNN")
    plt.xlabel("Value of k")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ========================================================
    # A9 - WEIGHTED KNN COMPARISON
    # ========================================================
    print("\n" + "=" * 60)
    print("A9: WEIGHTED KNN COMPARISON")
    print("=" * 60)

    weighted_accuracies = []
    weighted_times = []
    for k_value in k_values:
        weighted_model_k = WeightedKNN(k=k_value, algorithm=sorting_algorithm)
        start_time = time.perf_counter()
        weighted_model_k.fit(X_train, y_train)
        weighted_accuracy_k = weighted_model_k.score(X_test, y_test)
        weighted_time_k = time.perf_counter() - start_time
        weighted_accuracies.append(weighted_accuracy_k)
        weighted_times.append(weighted_time_k)

    # A9 RESULT TABLE
    print("\nWeighted KNN Results")
    print("{:<8}{:<20}{:<20}{:<20}".format("K", "Custom KNN", "Weighted KNN", "Scikit-learn"))
    for i in range(len(k_values)):
        print("{:<8}{:<20}{:<20}{:<20}".format(
            k_values[i], round(custom_accuracies[i], 4), round(weighted_accuracies[i], 4),
            round(sklearn_accuracies[i], 4)))

    # BEST WEIGHTED K
    best_weighted_index = np.argmax(weighted_accuracies)
    print("\nBest K for Weighted KNN:", k_values[best_weighted_index])
    print("Best Weighted KNN Accuracy:", round(weighted_accuracies[best_weighted_index] * 100, 2), "%")

    # A9 ACCURACY GRAPH
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, custom_accuracies, marker="o", label="Custom KNN")
    plt.plot(k_values, weighted_accuracies, marker="^", label="Weighted KNN")
    plt.plot(k_values, sklearn_accuracies, marker="s", label="Scikit-learn KNN")
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: KNN and Weighted KNN")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # A9 TIME GRAPH
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, custom_times, marker="o", label="Custom KNN")
    plt.plot(k_values, weighted_times, marker="^", label="Weighted KNN")
    plt.plot(k_values, sklearn_times, marker="s", label="Scikit-learn KNN")
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
    print("\nNumber of identical predictions:", np.sum(prediction_comparison))
    print("Total test samples:", len(y_test))
    print("Predictions identical:", np.all(prediction_comparison))

    # ========================================================
    # A10 - THREE-WAY PERFORMANCE COMPARISON (Q10, final question)
    #   1. Own KNN code           -> CustomKNN
    #   2. Scikit-learn's KNN     -> KNeighborsClassifier
    #   3. AI-optimized KNN code  -> OptimizedKNN
    # For each: Accuracy, Precision, Recall, F1-score, and computational
    # time averaged over 10 runs.
    # ========================================================
    print("\n" + "=" * 60)
    print("A10: THREE-WAY PERFORMANCE COMPARISON")
    print("=" * 60)

    comparison_k = 3
    n_runs = 10

    def time_fit_predict(model_factory, n_runs=n_runs):
        # Builds and trains a fresh model each run so timing includes both
        # fit() and predict(), then averages over n_runs for a stable number.
        run_times = []
        predictions = None
        for _ in range(n_runs):
            model = model_factory()
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            run_times.append(time.perf_counter() - start_time)
        return predictions, calculate_mean(run_times)

    custom_preds_q10, custom_avg_time = time_fit_predict(
        lambda: CustomKNN(k=comparison_k, algorithm="selection")
    )
    sklearn_preds_q10, sklearn_avg_time = time_fit_predict(
        lambda: KNeighborsClassifier(n_neighbors=comparison_k)
    )
    optimized_preds_q10, optimized_avg_time = time_fit_predict(
        lambda: OptimizedKNN(k=comparison_k)
    )

    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

    results = {
        "Own KNN (CustomKNN)": {**compute_metrics(y_test, custom_preds_q10), "time": custom_avg_time},
        "Scikit-learn KNN": {**compute_metrics(y_test, sklearn_preds_q10), "time": sklearn_avg_time},
        "AI-Optimized KNN": {**compute_metrics(y_test, optimized_preds_q10), "time": optimized_avg_time},
    }

    print(f"\n(k = {comparison_k}, time averaged over {n_runs} runs of fit + predict)\n")
    header = "{:<22}{:<12}{:<12}{:<10}{:<10}{:<16}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1", "Avg Time (s)"
    )
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print("{:<22}{:<12}{:<12}{:<10}{:<10}{:<16}".format(
            model_name,
            round(metrics["accuracy"], 4),
            round(metrics["precision"], 4),
            round(metrics["recall"], 4),
            round(metrics["f1"], 4),
            round(metrics["time"], 6),
        ))

    fastest_model = min(results, key=lambda name: results[name]["time"])
    slowest_model = max(results, key=lambda name: results[name]["time"])
    print(f"\nFastest: {fastest_model} ({round(results[fastest_model]['time'], 6)} s avg)")
    print(f"Slowest: {slowest_model} ({round(results[slowest_model]['time'], 6)} s avg)")
    speedup = results["Own KNN (CustomKNN)"]["time"] / results["AI-Optimized KNN"]["time"]
    print(f"AI-Optimized KNN is {round(speedup, 2)}x faster than Own KNN (CustomKNN) on this dataset/k.")

    # A10 - TIME COMPARISON BAR CHART
    plt.figure(figsize=(7, 5))
    model_names = list(results.keys())
    avg_times = [results[name]["time"] for name in model_names]
    plt.bar(model_names, avg_times, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylabel("Average Execution Time (seconds)")
    plt.title(f"A10: Avg Time over {n_runs} runs (fit + predict), k={comparison_k}")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()