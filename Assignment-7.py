import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# A4 - BINNING FUNCTIONS

def equal_width_binning(values, number_of_bins=4):
    values = np.asarray(values, dtype=float)
    minimum = np.min(values)
    maximum = np.max(values)
    if minimum == maximum:
        return np.zeros(len(values), dtype=int)
    bin_edges = np.linspace(minimum, maximum, number_of_bins + 1)
    binned_values = np.digitize(values, bin_edges[1:-1])
    return binned_values

def frequency_binning(values, number_of_bins=4):
    values = np.asarray(values, dtype=float)
    sorted_indices = np.argsort(values)
    binned_values = np.zeros(len(values), dtype=int)
    for bin_number, indices in enumerate(np.array_split(sorted_indices, number_of_bins)):binned_values[indices] = bin_number
    return binned_values

def bin_feature(values, binning_type="equal_width", number_of_bins=4):
    if binning_type == "equal_width":
        return equal_width_binning(values, number_of_bins)
    if binning_type == "frequency":
        return frequency_binning(values, number_of_bins)
    raise ValueError("Binning type must be 'equal_width' or 'frequency'")


# A1 - ENTROPY
def calculate_entropy(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = 0.0
    for probability in probabilities:
        if probability > 0:
            entropy -= probability * np.log2(probability)
    return entropy

# A2 - GINI INDEX
def calculate_gini(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1.0
    for probability in probabilities:
        gini -= probability ** 2
    return gini

# A3 - INFORMATION GAIN
def calculate_information_gain(feature_values, labels):
    feature_values = np.asarray(feature_values)
    labels = np.asarray(labels)
    parent_entropy = calculate_entropy(labels)
    weighted_entropy = 0.0
    unique_values = np.unique(feature_values)
    for value in unique_values:
        subset_labels = labels[feature_values == value]
        weight = len(subset_labels) / len(labels)
        weighted_entropy += weight * calculate_entropy(subset_labels)
    information_gain = parent_entropy - weighted_entropy
    return information_gain

def find_root_feature(features, labels, feature_names):
    information_gains = []
    for column in range(features.shape[1]):
        gain = calculate_information_gain(features[:, column],labels)
        information_gains.append(gain)
    best_feature_index = np.argmax(information_gains)
    return (best_feature_index,feature_names[best_feature_index],information_gains)

# A5 - OWN DECISION TREE
class DecisionTreeNode:
    def __init__(self,feature_index=None,feature_name=None,children=None,prediction=None,depth=0):
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.children = children if children is not None else {}
        self.prediction = prediction
        self.depth = depth
class CustomDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
    def fit(self, features, labels, feature_names):
        self.feature_names = feature_names
        available_features = list(range(features.shape[1]))
        self.root = self._build_tree(features,labels,available_features,depth=0)
    def _build_tree(self,features,labels,available_features,depth):
        majority_class = self._majority_class(labels)
        if len(np.unique(labels)) == 1:
            return DecisionTreeNode(prediction=majority_class,depth=depth)
        if depth >= self.max_depth:
            return DecisionTreeNode(prediction=majority_class,depth=depth)
        if len(labels) < self.min_samples_split:
            return DecisionTreeNode(prediction=majority_class,depth=depth)
        if len(available_features) == 0:
            return DecisionTreeNode(prediction=majority_class,depth=depth)
        best_feature = None
        best_gain = -1
        for feature_index in available_features:
            gain = calculate_information_gain(features[:, feature_index],labels)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
        if best_feature is None or best_gain <= 0:
            return DecisionTreeNode(prediction=majority_class,depth=depth)
        node = DecisionTreeNode(feature_index=best_feature,feature_name=self.feature_names[best_feature],depth=depth)
        feature_values = features[:, best_feature]
        unique_values = np.unique(feature_values)
        remaining_features = [feature for feature in available_features
                              if feature != best_feature]
        for value in unique_values:
            mask = feature_values == value
            child_features = features[mask]
            child_labels = labels[mask]
            child_node = self._build_tree(child_features,child_labels,remaining_features,depth + 1)
            node.children[value] = child_node
        return node
    def _majority_class(self, labels):
        values, counts = np.unique(labels,return_counts=True)
        return values[np.argmax(counts)]
    def _predict_sample(self, sample, node):
        if node.prediction is not None:
            return node.prediction
        feature_value = sample[node.feature_index]
        if feature_value in node.children:
            return self._predict_sample(sample,node.children[feature_value])
        return 0
    def predict(self, features):
        predictions = []
        for sample in features:
            prediction = self._predict_sample(sample,self.root)
            predictions.append(prediction)
        return np.array(predictions)
    def score(self, features, labels):
        predictions = self.predict(features)
        return accuracy_score(labels, predictions)

# TREE VISUALIZATION FOR OUR CUSTOM TREE
def draw_custom_tree(node, x=0.5, y=1.0, horizontal_spacing=1.0,vertical_spacing=0.12, positions=None,labels=None, node_counter=None):
    if positions is None:
        positions = {}
    if labels is None:
        labels = {}
    if node_counter is None:
        node_counter = [0]
    current_id = node_counter[0]
    node_counter[0] += 1
    positions[current_id] = (x, y)
    if node.prediction is not None:
        labels[current_id] = "Class = " + str(node.prediction)
        return positions, labels, current_id
    labels[current_id] = node.feature_name
    child_items = list(node.children.items())
    number_of_children = len(child_items)
    for index, (value, child) in enumerate(child_items):
        child_x = (x+ (index - (number_of_children - 1) / 2)* horizontal_spacing)
        child_y = y - vertical_spacing
        positions, labels, child_id = draw_custom_tree(child,child_x,child_y,horizontal_spacing / max(number_of_children, 1),vertical_spacing,positions,labels,node_counter)
        plt.plot([x, child_x],[y, child_y],'k-')
        plt.text((x + child_x) / 2,(y + child_y) / 2,str(value),fontsize=8)
    return positions, labels, current_id

def visualize_custom_tree(tree):
    plt.figure(figsize=(18, 10))
    positions, labels, _ = draw_custom_tree(tree.root,x=0.5,y=1.0,horizontal_spacing=0.5,vertical_spacing=0.08)
    for node_id, position in positions.items():
        plt.scatter(position[0],position[1],s=800)
        plt.text(position[0],position[1],labels[node_id],ha="center",va="center",fontsize=7)
    plt.axis("off")
    plt.title("Custom Decision Tree")
    plt.show()

# A7 - DECISION BOUNDARY
def plot_decision_boundary(model,features,labels,feature_names):
    x_min = features[:, 0].min() - 0.5
    x_max = features[:, 0].max() + 0.5
    y_min = features[:, 1].min() - 0.5
    y_max = features[:, 1].max() + 0.5
    x_values = np.linspace(x_min, x_max, 200)
    y_values = np.linspace(y_min, y_max, 200)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.c_[xx.ravel(),yy.ravel()]
    predictions = model.predict(grid)
    predictions = predictions.reshape(xx.shape)
    plt.figure(figsize=(9, 7))
    plt.contourf(xx,yy,predictions,alpha=0.3)
    plt.scatter(features[:, 0],features[:, 1],c=labels,edgecolor="black")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Boundary using Two Features")
    plt.show()

# A8 - HYPERPARAMETER TUNING
def perform_grid_search(features, labels):
    model = DecisionTreeClassifier(random_state=42)
    parameter_grid = {"criterion": ["gini","entropy"],"max_depth": [2,3,4,5,None],"min_samples_split": [2,5,10],"min_samples_leaf": [1,2,5]}
    grid_search = GridSearchCV(estimator=model,param_grid=parameter_grid,cv=5,scoring="accuracy")
    grid_search.fit(features,labels)
    return grid_search


# MAIN PROGRAM
if __name__ == "__main__":
    digits = load_digits()
    features = digits.data
    labels = digits.target
    print("Original Dataset Shape:", features.shape)
    print("Original Number of Classes:", len(np.unique(labels)))
    selected_indices = np.isin(labels,[0, 1])
    features = features[selected_indices]
    labels = labels[selected_indices]
    print("\nSelected Dataset Shape:", features.shape)
    print("Selected Classes:", np.unique(labels))
    print("Class Distribution:",np.bincount(labels))

    # A4 - BINNING
    binned_features = np.zeros_like(features,dtype=int)
    for column in range(features.shape[1]):
        binned_features[:, column] = bin_feature(features[:, column],binning_type="equal_width",number_of_bins=4)
    feature_names = ["Pixel_" + str(index) for index in range(features.shape[1])]

    # A1 - ENTROPY
    entropy = calculate_entropy(labels)
    print("\nA1 - ENTROPY")
    print("Entropy of Dataset:", entropy)

    # A2 - GINI INDEX
    gini = calculate_gini(labels)
    print("\nA2 - GINI INDEX")
    print("Gini Index of Dataset:", gini)

    # A3 - ROOT NODE
    root_index, root_name, information_gains = find_root_feature(binned_features,labels,feature_names)
    print("\nA3 - ROOT FEATURE")
    print("Root Feature:", root_name)
    print("Root Feature Index:", root_index)
    print("Information Gain:", information_gains[root_index])
    gain_table = pd.DataFrame({"Feature": feature_names,"Information_Gain": information_gains})
    gain_table = gain_table.sort_values(by="Information_Gain",ascending=False)
    print("\nTop Information Gain Features:")
    print(gain_table.head(10))

    # A5 - BUILD OUR OWN DECISION TREE
    X_train, X_test, y_train, y_test = train_test_split(binned_features,labels,test_size=0.30,random_state=42,stratify=labels)
    custom_tree = CustomDecisionTree(max_depth=5)
    custom_tree.fit(X_train,y_train,feature_names)
    custom_predictions = custom_tree.predict(X_test)
    custom_accuracy = accuracy_score(y_test,custom_predictions)
    print("\nA5 - CUSTOM DECISION TREE")
    print("Custom Decision Tree Accuracy:",custom_accuracy)

    # A6 - VISUALIZE OUR DECISION TREE
    print("\nA6 - DECISION TREE VISUALIZATION")
    visualize_custom_tree(custom_tree)

    # A7 - TWO FEATURE DECISION BOUNDARY
    top_two_features = (gain_table.head(2).index.to_list())
    feature_1 = top_two_features[0]
    feature_2 = top_two_features[1]
    two_features = binned_features[:,[feature_1, feature_2]]
    two_feature_names = [feature_names[feature_1],feature_names[feature_2]]
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(two_features,labels,test_size=0.30,random_state=42,stratify=labels)
    two_feature_tree = CustomDecisionTree(max_depth=5)
    two_feature_tree.fit(X_train_2,y_train_2,two_feature_names)
    print("\nA7 - TWO FEATURE DECISION BOUNDARY")
    print("Selected Features:",two_feature_names)
    print("Two Feature Accuracy:",two_feature_tree.score(X_test_2,y_test_2))
    plot_decision_boundary(two_feature_tree,two_features,labels,two_feature_names)

    # A8 - GRID SEARCH
    print("\nA8 - HYPERPARAMETER TUNING")
    grid_search = perform_grid_search(features,labels)
    print("Best Parameters:",grid_search.best_params_)
    print("Best Cross-Validation Accuracy:",grid_search.best_score_)
    best_model = grid_search.best_estimator_
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(features,labels,test_size=0.30,random_state=42,stratify=labels)
    best_model.fit(X_train_original,y_train_original)
    final_predictions = best_model.predict(X_test_original)
    print("\nFinal Tuned Decision Tree Results")
    print("Accuracy:",accuracy_score(y_test_original,final_predictions))
    print("Precision:",precision_score(y_test_original,final_predictions))
    print("Recall:",recall_score(y_test_original,final_predictions))
    print("F1 Score:",f1_score(y_test_original,final_predictions))
    plt.figure(figsize=(20, 10))
    plot_tree(best_model,max_depth=3,filled=True,feature_names=feature_names,class_names=["0", "1"],fontsize=8)
    plt.title("Tuned Decision Tree - Digits Dataset")
    plt.show()