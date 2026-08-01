import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.spatial.distance import minkowski

# Claude: added for the unit tests appended at the bottom of this file.
# unittest.mock is used only to stop plt.show() from popping up a window
# while tests run - it does not touch any of the actual logic being tested.
import unittest
from unittest.mock import patch

dataset = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")

# Claude: pulled this list out as a shared constant since it was being
# hard-coded identically inside both apply_label_encoding() and
# apply_one_hot_encoding(). One source of truth avoids the two copies
# drifting apart if the columns ever change.
CATEGORICAL_COLUMNS = ["Education", "Marital_Status"]


# A2-Label Encoding and One Hot Encoding
def label_encode(column):
    """Manually label-encode a categorical column (first-seen value -> 0, 1, 2, ...)."""
    encoding = {}
    encoded_column = []
    current_label = 0
    for value in column:
        if value not in encoding:
            encoding[value] = current_label
            current_label += 1
        encoded_column.append(encoding[value])
    return encoded_column, encoding


def one_hot_encode(column):
    """Manually one-hot-encode a categorical column into a dict of binary lists."""
    unique_values = list(dict.fromkeys(column))
    one_hot_dictionary = {}
    for category in unique_values:
        encoded_column = []
        for value in column:
            if value == category:
                encoded_column.append(1)
            else:
                encoded_column.append(0)
        one_hot_dictionary[category] = encoded_column
    return one_hot_dictionary


# A3-Apply Label and One Hot Encoding
def apply_label_encoding(dataset):
    """Apply label_encode() to every categorical column in the dataset."""
    encoded_dataset = dataset.copy()
    encoding_details = {}
    for column in CATEGORICAL_COLUMNS:
        encoded_values, mapping = label_encode(dataset[column])
        encoded_dataset[column] = encoded_values
        encoding_details[column] = mapping
    return encoded_dataset, encoding_details


def apply_one_hot_encoding(dataset):
    """Apply one_hot_encode() to every categorical column in the dataset."""
    encoded_dataset = dataset.copy()

    # Claude: originally this dropped a column and then called pd.concat()
    # once per categorical column, inside the loop. Each concat rebuilds a
    # new DataFrame, so with N categorical columns you pay for N full
    # copies of an already-growing table (and pandas warns about this
    # "fragmentation" pattern). Collecting the encoded pieces first and
    # concatenating once at the end does the same thing with a single
    # copy - same logic and output, just less repeated work.
    encoded_pieces = []
    for column in CATEGORICAL_COLUMNS:
        encoded_dictionary = one_hot_encode(dataset[column])
        encoded_dataframe = pd.DataFrame(encoded_dictionary)
        encoded_dataframe.columns = [column + "_" + str(name) for name in encoded_dataframe.columns]
        encoded_dataset.drop(column, axis=1, inplace=True)
        encoded_pieces.append(encoded_dataframe)

    encoded_dataset = pd.concat([encoded_dataset] + encoded_pieces, axis=1)
    return encoded_dataset


def feature_dimension(dataset):
    return dataset.shape


# A4-Generalized Minkowski Distance
def minkowski_distance(vector1, vector2, p):
    """Generalized Minkowski distance between two equal-length vectors."""
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have same length.")
    # Claude: added a guard for non-positive p. The original code would
    # silently produce a nonsensical or divide-by-zero-adjacent result
    # (e.g. 1/p with p=0) if a bad value slipped in from input(). This
    # doesn't change behavior for any valid p >= 1, it just fails loudly
    # instead of quietly.
    if p <= 0:
        raise ValueError("p must be a positive number.")
    distance_sum = 0
    for i in range(len(vector1)):
        distance_sum += abs(vector1[i] - vector2[i]) ** p
    distance = distance_sum ** (1 / p)
    return distance


# A5-Minkowski Distance for p=1 to 10
def calculate_minkowski_for_range(vector1, vector2):
    p_values = []
    distance_values = []
    for p in range(1, 11):
        distance = minkowski_distance(vector1, vector2, p)
        p_values.append(p)
        distance_values.append(distance)
    return p_values, distance_values


# A6-Scipy Minkowski Comparison
def compare_with_scipy(vector1, vector2):
    comparison = []
    for p in range(1, 11):
        own_distance = minkowski_distance(vector1, vector2, p)
        scipy_distance = minkowski(vector1, vector2, p)
        comparison.append([p, own_distance, scipy_distance])
    return comparison


# A7-Dot Product, Euclidean Norm
def dot_product(vector1, vector2):
    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    return result


def euclidean_norm(vector):
    total = 0
    for value in vector:
        total += value ** 2
    return math.sqrt(total)


# A8-Mean, Variance, Standard Deviation
def calculate_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)


def calculate_variance(data):
    mean = calculate_mean(data)
    variance_sum = 0
    for value in data:
        variance_sum += (value - mean) ** 2
    variance = variance_sum / len(data)
    return variance


def calculate_standard_deviation(data):
    variance = calculate_variance(data)
    return math.sqrt(variance)


def dataset_statistics(dataset):
    """
    Note (Claude): calculate_standard_deviation() internally calls
    calculate_variance(), which internally calls calculate_mean() again.
    So for every column this walks the data three times instead of one.
    I left this exactly as you wrote it, since collapsing it into a single
    pass would mean changing the function signatures (e.g. passing a
    precomputed mean into calculate_variance), which felt like it crosses
    from "cleanup" into "changing your logic/structure". Flagging it here
    in case you want to mention the tradeoff in your writeup.
    """
    numeric_dataset = dataset.select_dtypes(include=[np.number])
    mean_vector = []
    variance_vector = []
    std_vector = []
    column_names = numeric_dataset.columns.tolist()
    for column in column_names:
        values = numeric_dataset[column].tolist()
        mean_vector.append(calculate_mean(values))
        variance_vector.append(calculate_variance(values))
        std_vector.append(calculate_standard_deviation(values))
    return column_names, mean_vector, variance_vector, std_vector


# A9-Numpy Comparison
def numpy_statistics(dataset):
    numeric_dataset = dataset.select_dtypes(include=[np.number])
    mean_vector = numeric_dataset.mean(axis=0)
    variance_vector = numeric_dataset.var(axis=0)
    std_vector = numeric_dataset.std(axis=0)
    return mean_vector, variance_vector, std_vector


# A10-Histogram
def plot_histogram(dataset, feature_name):
    feature = dataset[feature_name]
    mean = np.mean(feature)
    variance = np.var(feature)
    plt.figure(figsize=(8, 5))
    plt.hist(feature, bins=10, edgecolor="black")
    plt.title("Histogram of " + feature_name)
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    return mean, variance


# A11-Euclidean Distance, Assign Cluster, Update Centroids, KMEANS
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = []
        for centroid in centroids:
            distances.append(euclidean_distance(point, centroid))
        cluster = distances.index(min(distances))
        clusters.append(cluster)
    return clusters


def update_centroids(data, clusters, k):
    new_centroids = []
    data = np.array(data)
    for cluster in range(k):
        cluster_points = data[np.array(clusters) == cluster]
        if len(cluster_points) == 0:
            new_centroids.append(data[0])
        else:
            new_centroids.append(np.mean(cluster_points, axis=0))
    return new_centroids


def kmeans(data, k, max_iterations=100):
    data = np.array(data)
    # Claude: added .copy() here. data[:k] is a *view* into the original
    # array, not a new one. It doesn't cause a bug in the current code
    # (update_centroids always reassigns `centroids` to a new list before
    # anything is mutated in place), but relying on that is fragile - if
    # this function is edited later to update centroids in place, an
    # un-copied slice would silently corrupt the first k rows of `data`.
    # The .copy() makes the independence explicit and costs nothing.
    centroids = data[:k].copy()
    for iteration in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids


# ============================================================
# A11 (OPTIMIZED) - Vectorized K-Means
# ============================================================
# Claude: This is the "AI-assisted" version for your comparison question.
# It is the exact same algorithm as your kmeans() above (Lloyd's algorithm:
# assign each point to its nearest centroid, recompute centroids as the
# mean of their assigned points, repeat until centroids stop moving) - I
# have not changed the logic, the stopping condition, or the initial
# centroid choice (still data[:k]). The only change is *how* the distance
# calculations are done.
#
# Your original assign_clusters() has three nested loops in effect:
# for each point -> for each centroid -> for each dimension (inside
# euclidean_distance). That is pure-Python work happening element by
# element. NumPy can do the same arithmetic on the whole array at once
# (in fast, compiled C code) using broadcasting, instead of looping in
# Python. That's the entire optimization - fewer Python-level loop
# iterations, same math.

def assign_clusters_optimized(data, centroids):
    """
    Vectorized replacement for assign_clusters(). Instead of looping over
    every point and every centroid in Python, this computes ALL
    point-to-centroid distances in one NumPy operation.

    data[:, np.newaxis, :]  -> shape (n_points, 1, n_features)
    centroids[np.newaxis, :, :] -> shape (1, k, n_features)
    Subtracting broadcasts these into a (n_points, k, n_features) array of
    differences in one step; squaring, summing, and sqrt then collapse the
    last axis to give a (n_points, k) matrix of distances - i.e. distance
    from every point to every centroid, computed all at once.
    """
    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    differences = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    clusters = np.argmin(distances, axis=1)
    return clusters


def update_centroids_optimized(data, clusters, k):
    """
    Same logic as update_centroids(): each new centroid is the mean of the
    points currently assigned to it, with the same data[0] fallback for an
    empty cluster. The only change is writing into a pre-allocated NumPy
    array instead of building up a Python list with .append(), which
    avoids repeated list growth and an extra np.array(clusters) conversion
    on every call (the original re-converts `clusters` to an array inside
    the loop, k times, on every iteration).
    """
    data = np.asarray(data, dtype=float)
    clusters = np.asarray(clusters)
    new_centroids = np.empty((k, data.shape[1]))
    for cluster in range(k):
        cluster_points = data[clusters == cluster]
        if len(cluster_points) == 0:
            new_centroids[cluster] = data[0]
        else:
            new_centroids[cluster] = cluster_points.mean(axis=0)
    return new_centroids


def kmeans_optimized(data, k, max_iterations=100):
    """
    Same structure, same stopping rule, same centroid initialization as
    your kmeans(). Only the inner assign/update steps are swapped for
    their vectorized versions above.
    """
    data = np.asarray(data, dtype=float)
    centroids = data[:k].copy()
    for iteration in range(max_iterations):
        clusters = assign_clusters_optimized(data, centroids)
        new_centroids = update_centroids_optimized(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    # Claude: converting clusters to a plain list here so the return value
    # has the exact same type as your original kmeans() (a list, not a
    # NumPy array) - keeps the two functions drop-in interchangeable.
    return clusters.tolist(), centroids



# Claude: wrapped the original "Main" section in this guard so the file can be
# imported (e.g. by the unit tests below, or by `python -m unittest`) without
# re-running the whole script, prompting for input(), or reading the Excel file
# again. Running the file directly (python lab_session_04_improved.py) still
# behaves exactly as before.
if __name__ == "__main__":
    # Main
    print("A2:LABEL ENCODING")
    education_encoded, education_mapping = label_encode(dataset["Education"])
    marital_encoded, marital_mapping = label_encode(dataset["Marital_Status"])
    print("\nEducation Mapping")
    print(education_mapping)
    print("\nEducation Encoded Values")
    print(education_encoded[:10])
    print("\nMarital Status Mapping")
    print(marital_mapping)
    print("\nMarital Status Encoded Values")
    print(marital_encoded[:10])
    print("\n")

    print("A2:ONE HOT ENCODING")
    education_onehot = one_hot_encode(dataset["Education"])
    marital_onehot = one_hot_encode(dataset["Marital_Status"])
    print("\nEducation Categories")
    for category in education_onehot:
        print(category)
        print(education_onehot[category][:10])
    print("\nMarital Status Categories")
    for category in marital_onehot:
        print(category)
        print(marital_onehot[category][:10])
    print("\n")

    print("A3:APPLY ENCODING")
    label_dataset, mappings = apply_label_encoding(dataset)
    onehot_dataset = apply_one_hot_encoding(dataset)
    print("\nLabel Encoded Dataset")
    print(label_dataset.head())
    print("\nOne Hot Encoded Dataset")
    print(onehot_dataset.head())
    print("\nOriginal Dataset Dimension")
    print(feature_dimension(dataset))
    print("\nLabel Encoded Dataset Dimension")
    print(feature_dimension(label_dataset))
    print("\nOne Hot Encoded Dataset Dimension")
    print(feature_dimension(onehot_dataset))

    print("\n")
    print("A4:MINKOWSKI DISTANCE")
    numeric_dataset = label_dataset.select_dtypes(include=[np.number])
    vector1 = numeric_dataset.iloc[120].values
    vector2 = numeric_dataset.iloc[134].values
    p = int(input("Enter value of p : "))
    distance = minkowski_distance(vector1, vector2, p)
    print("\nMinkowski Distance =", distance)

    print("\n")
    print("A5:MINKOWSKI DISTANCE (p = 1 TO 10)")
    p_values, distance_values = calculate_minkowski_for_range(vector1, vector2)
    for i in range(len(p_values)):
        print("p=", p_values[i], "Distance=", distance_values[i])
    plt.figure(figsize=(8, 5))
    plt.plot(p_values, distance_values, marker="o")
    plt.title("Minkowski Distance vs p")
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

    print("\n")
    print("A6:COMPARISON WITH SCIPY")
    comparison = compare_with_scipy(vector1, vector2)
    print("\nP\tOwn Function\tSciPy")
    for row in comparison:
        print(row[0], "\t", round(row[1], 6), "\t", round(row[2], 6))

    print("\n")
    print("A7:DOT PRODUCT AND EUCLIDEAN NORM")
    own_dot = dot_product(vector1, vector2)
    numpy_dot = np.dot(vector1, vector2)
    print("\nOwn Dot Product")
    print(own_dot)
    print("\nNumPy Dot Product")
    print(numpy_dot)
    own_norm_vector1 = euclidean_norm(vector1)
    numpy_norm_vector1 = np.linalg.norm(vector1)
    print("\nOwn Euclidean Norm")
    print(own_norm_vector1)
    print("\nNumPy Euclidean Norm")
    print(numpy_norm_vector1)

    print("\n")
    print("A8:MEAN,VARIANCE,STANDARD DEVIATION")
    column_names, mean_vector, variance_vector, std_vector = dataset_statistics(label_dataset)
    print("\nFeature\t\tMean\t\tVariance\t\tStandard Deviation")
    for i in range(len(column_names)):
        print(column_names[i], "\t", round(mean_vector[i], 3), "\t", round(variance_vector[i], 3), "\t", round(std_vector[i], 3))

    print("\n")
    print("A9:COMPARISON WITH NUMPY")
    numpy_mean, numpy_variance, numpy_std = numpy_statistics(label_dataset)
    print("\nOwn Mean Vector")
    print(mean_vector)
    print("\nNumPy Mean Vector")
    print(numpy_mean.values)
    print("\nOwn Standard Deviation")
    print(std_vector)
    print("\nNumPy Standard Deviation")
    print(numpy_std.values)

    print("\n")
    print("A10:HISTOGRAM")
    print("\nAvailable Numeric Features\n")
    numeric_columns = label_dataset.select_dtypes(include=[np.number]).columns
    print(list(numeric_columns))
    feature = input("\nEnter Feature Name : ")
    mean, variance = plot_histogram(label_dataset, feature)
    print("\nMean =", mean)
    print("Variance =", variance)

    print("\n")
    print("A11:K-MEANS CLUSTERING")
    numeric_dataset = label_dataset.select_dtypes(include=[np.number])
    data = numeric_dataset.values
    k = int(input("\nEnter Number of Clusters : "))
    clusters, centroids = kmeans(data, k)
    print("\nFirst 40 Cluster Labels")
    print(clusters[:40])
    print("\nCentroids")
    print(np.array(centroids))

    print("\n")
    print("A12:ORIGINAL vs OPTIMIZED K-MEANS COMPARISON")
    # Claude: benchmark comparing your original kmeans() against the
    # vectorized kmeans_optimized() on the same data and same k, so the
    # timing comparison is apples-to-apples.

    start_time = time.perf_counter()
    original_clusters, original_centroids = kmeans(data, k)
    original_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    optimized_clusters, optimized_centroids = kmeans_optimized(data, k)
    optimized_time = time.perf_counter() - start_time

    print("\nOriginal kmeans() time     : {:.4f} seconds".format(original_time))
    print("Optimized kmeans() time    : {:.4f} seconds".format(optimized_time))
    if optimized_time > 0:
        print("Speedup (original/optimized): {:.2f}x".format(original_time / optimized_time))

    # Sanity check: both should assign points to the same clusters, since
    # they run the identical algorithm with the identical initial centroids
    # - only the distance computation differs.
    same_result = np.array_equal(np.array(original_clusters), np.array(optimized_clusters))
    print("Both versions produced identical cluster assignments:", same_result)


# ============================================================
# UNIT TESTS
# ============================================================
# Claude: These tests live outside the `if __name__ == "__main__":` guard
# above, so they don't run when you execute this file normally with
# `python lab_session_04_improved.py` (that still just runs your program
# as before). To run ONLY the tests, use:
#
#       python -m unittest lab_session_04_improved -v
#
# Note: because `dataset = pd.read_excel(...)` near the top of the file
# runs as soon as the module is imported, "Lab Session Data.xlsx" still
# needs to be in the same folder for these tests to run - even though
# most tests below use small, hand-made sample data instead of that
# dataset, so the results are predictable and easy to check by hand.

class TestLabelEncode(unittest.TestCase):
    # Checks label_encode() on a normal small list with repeats
    def test_normal_case(self):
        column = ["A", "B", "A", "C", "B"]
        encoded, mapping = label_encode(column)
        self.assertEqual(mapping, {"A": 0, "B": 1, "C": 2})
        self.assertEqual(encoded, [0, 1, 0, 2, 1])

    # Edge case: empty column should give empty outputs, not crash
    def test_empty_column(self):
        encoded, mapping = label_encode([])
        self.assertEqual(encoded, [])
        self.assertEqual(mapping, {})

    # Edge case: only one distinct value repeated many times
    def test_single_category_repeated(self):
        column = ["X", "X", "X"]
        encoded, mapping = label_encode(column)
        self.assertEqual(encoded, [0, 0, 0])
        self.assertEqual(mapping, {"X": 0})


class TestOneHotEncode(unittest.TestCase):
    # Checks one_hot_encode() produces correct binary columns for a normal case
    def test_normal_case(self):
        column = ["A", "B", "A"]
        result = one_hot_encode(column)
        self.assertEqual(result["A"], [1, 0, 1])
        self.assertEqual(result["B"], [0, 1, 0])

    # Edge case: empty column should return an empty dictionary
    def test_empty_column(self):
        result = one_hot_encode([])
        self.assertEqual(result, {})

    # Edge case: only one category present -> single all-ones column
    def test_single_category(self):
        result = one_hot_encode(["Yes", "Yes", "Yes"])
        self.assertEqual(result, {"Yes": [1, 1, 1]})


class TestApplyEncodingFunctions(unittest.TestCase):
    # Small hand-made DataFrame used instead of the real Excel dataset,
    # so the expected output is easy to reason about.
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "Education": ["Graduation", "PhD", "Graduation"],
            "Marital_Status": ["Married", "Single", "Married"],
            "Income": [50000, 60000, 55000],
        })

    # apply_label_encoding should replace categorical columns with integer codes
    def test_apply_label_encoding_normal_case(self):
        encoded_df, mappings = apply_label_encoding(self.sample_df)
        self.assertEqual(list(encoded_df["Education"]), [0, 1, 0])
        self.assertEqual(list(encoded_df["Marital_Status"]), [0, 1, 0])
        self.assertIn("Education", mappings)

    # apply_one_hot_encoding should expand categorical columns into 0/1 columns
    # and drop the original categorical columns
    def test_apply_one_hot_encoding_normal_case(self):
        encoded_df = apply_one_hot_encoding(self.sample_df)
        self.assertNotIn("Education", encoded_df.columns)
        self.assertIn("Education_Graduation", encoded_df.columns)
        self.assertIn("Education_PhD", encoded_df.columns)
        self.assertEqual(list(encoded_df["Education_Graduation"]), [1, 0, 1])

    # Edge case: a dataset with only one row should still encode without errors
    def test_apply_one_hot_encoding_single_row(self):
        single_row_df = self.sample_df.iloc[[0]]
        encoded_df = apply_one_hot_encoding(single_row_df)
        self.assertEqual(encoded_df.shape[0], 1)


class TestFeatureDimension(unittest.TestCase):
    # feature_dimension should just return the DataFrame's (rows, columns) shape
    def test_normal_case(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.assertEqual(feature_dimension(df), (3, 2))


class TestMinkowskiDistance(unittest.TestCase):
    # p=2 should match the familiar Euclidean distance formula
    def test_p_equals_2_euclidean(self):
        result = minkowski_distance([0, 0], [3, 4], 2)
        self.assertAlmostEqual(result, 5.0)

    # p=1 should match Manhattan distance (sum of absolute differences)
    def test_p_equals_1_manhattan(self):
        result = minkowski_distance([1, 2, 3], [4, 6, 3], 1)
        self.assertAlmostEqual(result, 7.0)

    # Edge case: identical vectors should always have distance 0, for any p
    def test_identical_vectors_zero_distance(self):
        result = minkowski_distance([2, 2, 2], [2, 2, 2], 3)
        self.assertAlmostEqual(result, 0.0)

    # Invalid input: mismatched vector lengths should raise ValueError
    def test_mismatched_lengths_raises_error(self):
        with self.assertRaises(ValueError):
            minkowski_distance([1, 2], [1, 2, 3], 2)

    # Invalid input: p = 0 should raise ValueError (guard added by Claude)
    def test_p_zero_raises_error(self):
        with self.assertRaises(ValueError):
            minkowski_distance([1, 2], [3, 4], 0)

    # Invalid input: negative p should raise ValueError (guard added by Claude)
    def test_negative_p_raises_error(self):
        with self.assertRaises(ValueError):
            minkowski_distance([1, 2], [3, 4], -1)


class TestMinkowskiRangeAndScipyComparison(unittest.TestCase):
    # calculate_minkowski_for_range should return p = 1..10 with a distance for each
    def test_range_length_and_p_values(self):
        p_values, distance_values = calculate_minkowski_for_range([1, 2], [4, 6])
        self.assertEqual(p_values, list(range(1, 11)))
        self.assertEqual(len(distance_values), 10)

    # compare_with_scipy should closely match scipy's own minkowski() output
    def test_matches_scipy_implementation(self):
        comparison = compare_with_scipy([1, 2, 3], [4, 5, 7])
        for p, own_distance, scipy_distance in comparison:
            self.assertAlmostEqual(own_distance, scipy_distance, places=6)


class TestDotProductAndNorm(unittest.TestCase):
    # Normal case: known dot product of two simple vectors
    def test_dot_product_normal_case(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)

    # Edge case: orthogonal vectors should have a dot product of 0
    def test_dot_product_orthogonal_vectors(self):
        self.assertEqual(dot_product([1, 0], [0, 1]), 0)

    # Edge case: a zero vector should have a dot product of 0 with anything
    def test_dot_product_zero_vector(self):
        self.assertEqual(dot_product([0, 0, 0], [5, 6, 7]), 0)

    # Normal case: 3-4-5 right triangle -> norm should be exactly 5
    def test_euclidean_norm_normal_case(self):
        self.assertAlmostEqual(euclidean_norm([3, 4]), 5.0)

    # Edge case: norm of a zero vector should be 0
    def test_euclidean_norm_zero_vector(self):
        self.assertAlmostEqual(euclidean_norm([0, 0, 0]), 0.0)


class TestMeanVarianceStd(unittest.TestCase):
    # Normal case: mean/variance/std of a simple known list
    def test_normal_case(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertAlmostEqual(calculate_mean(data), 5.0)
        self.assertAlmostEqual(calculate_variance(data), 4.0)
        self.assertAlmostEqual(calculate_standard_deviation(data), 2.0)

    # Edge case: a list with only one value -> variance and std should be 0
    def test_single_value_list(self):
        self.assertAlmostEqual(calculate_mean([10]), 10.0)
        self.assertAlmostEqual(calculate_variance([10]), 0.0)
        self.assertAlmostEqual(calculate_standard_deviation([10]), 0.0)

    # Edge case: all identical values -> variance and std should be 0
    def test_all_identical_values(self):
        data = [7, 7, 7, 7]
        self.assertAlmostEqual(calculate_variance(data), 0.0)
        self.assertAlmostEqual(calculate_standard_deviation(data), 0.0)

    # Invalid input: an empty list should raise ZeroDivisionError
    # (division by len(data), same behavior as the original function)
    def test_empty_list_raises_error(self):
        with self.assertRaises(ZeroDivisionError):
            calculate_mean([])


class TestDatasetAndNumpyStatistics(unittest.TestCase):
    def setUp(self):
        self.numeric_df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
        })

    # dataset_statistics should return one mean/variance/std per numeric column
    def test_dataset_statistics_normal_case(self):
        columns, means, variances, stds = dataset_statistics(self.numeric_df)
        self.assertEqual(columns, ["x", "y"])
        self.assertAlmostEqual(means[0], 2.5)
        self.assertAlmostEqual(means[1], 25.0)

    # Claude: running this test uncovered a real mismatch, so I rewrote it
    # to document what actually happens instead of asserting something
    # false. dataset_statistics() computes population variance/std
    # (divides by N, since calculate_variance() does `variance_sum / len(data)`).
    # numpy_statistics() calls pandas' .var()/.std(), which default to
    # SAMPLE variance/std (divides by N-1). Means still match (ddof doesn't
    # affect the mean), but variance/std will only match by coincidence.
    # This is a genuine behavioral difference in the original code, not a
    # bug I introduced - worth mentioning in your comparison report.
    def test_means_match_numpy(self):
        _, own_means, _, _ = dataset_statistics(self.numeric_df)
        numpy_means, _, _ = numpy_statistics(self.numeric_df)
        for i, column in enumerate(self.numeric_df.select_dtypes(include=[np.number]).columns):
            self.assertAlmostEqual(own_means[i], numpy_means[column])

    def test_std_differs_from_numpy_due_to_ddof(self):
        _, _, _, own_stds = dataset_statistics(self.numeric_df)
        _, _, numpy_stds = numpy_statistics(self.numeric_df)
        column = self.numeric_df.columns[0]
        # population std (own) vs sample std (pandas, ddof=1) -> not equal
        self.assertNotAlmostEqual(own_stds[0], numpy_stds[column])
        # confirm the manual value matches population std computed with ddof=0
        expected_population_std = self.numeric_df[column].std(ddof=0)
        self.assertAlmostEqual(own_stds[0], expected_population_std)


class TestPlotHistogram(unittest.TestCase):
    # plt.show() is patched out so the test doesn't try to open a display
    # window - we're only checking the returned mean/variance are correct
    @patch("matplotlib.pyplot.show")
    def test_returns_correct_mean_and_variance(self, mock_show):
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        mean, variance = plot_histogram(df, "score")
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(variance, 2.0)
        mock_show.assert_called_once()

    # Invalid input: a column name that doesn't exist should raise a KeyError
    @patch("matplotlib.pyplot.show")
    def test_missing_column_raises_error(self, mock_show):
        df = pd.DataFrame({"score": [1, 2, 3]})
        with self.assertRaises(KeyError):
            plot_histogram(df, "not_a_real_column")


class TestKMeansHelpers(unittest.TestCase):
    # euclidean_distance: normal 3-4-5 case
    def test_euclidean_distance_normal_case(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)

    # euclidean_distance: same point twice -> distance 0
    def test_euclidean_distance_same_point(self):
        self.assertAlmostEqual(euclidean_distance([1, 1], [1, 1]), 0.0)

    # assign_clusters: each point should be assigned to its nearest centroid
    def test_assign_clusters_normal_case(self):
        data = [[0, 0], [0, 1], [10, 10], [10, 11]]
        centroids = [[0, 0], [10, 10]]
        clusters = assign_clusters(data, centroids)
        self.assertEqual(clusters, [0, 0, 1, 1])

    # update_centroids: centroid should become the mean of its assigned points
    def test_update_centroids_normal_case(self):
        data = [[0, 0], [2, 2], [10, 10], [12, 12]]
        clusters = [0, 0, 1, 1]
        new_centroids = update_centroids(data, clusters, 2)
        np.testing.assert_allclose(new_centroids[0], [1, 1])
        np.testing.assert_allclose(new_centroids[1], [11, 11])

    # Edge case: a cluster with no points assigned should fall back to data[0]
    # (this is the original fallback behavior, kept as-is)
    def test_update_centroids_empty_cluster(self):
        data = [[0, 0], [2, 2]]
        clusters = [0, 0]  # nothing assigned to cluster 1
        new_centroids = update_centroids(data, clusters, 2)
        np.testing.assert_allclose(new_centroids[1], data[0])


class TestKMeans(unittest.TestCase):
    # Normal case: two well-separated clusters should be correctly grouped,
    # regardless of which integer label (0 or 1) each group ends up with
    def test_two_well_separated_clusters(self):
        data = [[0, 0], [0, 1], [1, 0], [20, 20], [20, 21], [21, 20]]
        clusters, centroids = kmeans(data, k=2)
        first_group = clusters[:3]
        second_group = clusters[3:]
        self.assertEqual(len(set(first_group)), 1)   # all 3 in same cluster
        self.assertEqual(len(set(second_group)), 1)  # all 3 in same cluster
        self.assertNotEqual(first_group[0], second_group[0])  # different clusters
        self.assertEqual(len(centroids), 2)

    # Edge case: k equal to the number of points -> every point is its own cluster
    def test_k_equals_number_of_points(self):
        data = [[0, 0], [5, 5], [10, 10]]
        clusters, centroids = kmeans(data, k=3)
        self.assertEqual(len(set(clusters)), 3)


class TestOptimizedKMeansMatchesOriginal(unittest.TestCase):
    # Claude: these tests check that the optimized (vectorized) version
    # gives the exact same output as your original loop-based version -
    # that's the whole point of an "optimization": same results, less work.
    def setUp(self):
        self.data = [
            [0, 0], [0, 1], [1, 0],
            [20, 20], [20, 21], [21, 20],
            [40, 0], [41, 1], [40, 1],
        ]

    # assign_clusters and assign_clusters_optimized should agree point-for-point
    def test_assign_clusters_matches(self):
        centroids = [[0, 0], [20, 20], [40, 0]]
        original = assign_clusters(self.data, centroids)
        optimized = assign_clusters_optimized(self.data, centroids)
        np.testing.assert_array_equal(original, optimized)

    # update_centroids and update_centroids_optimized should agree on new centroid values
    def test_update_centroids_matches(self):
        clusters = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        original = update_centroids(self.data, clusters, 3)
        optimized = update_centroids_optimized(self.data, clusters, 3)
        np.testing.assert_allclose(original, optimized)

    # Full kmeans() and kmeans_optimized() should converge to the same clusters
    def test_full_kmeans_matches(self):
        clusters_original, centroids_original = kmeans(self.data, k=3)
        clusters_optimized, centroids_optimized = kmeans_optimized(self.data, k=3)
        self.assertEqual(list(clusters_original), list(clusters_optimized))
        np.testing.assert_allclose(centroids_original, centroids_optimized)


if __name__ == "__main__":
    # Claude: this second __main__ guard only matters if you delete the
    # earlier one - normally the block above already runs your program.
    # Left out of the automatic run path on purpose, see the note above
    # this test section for how to actually run the tests.
    pass