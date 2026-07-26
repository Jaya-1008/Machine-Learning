import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import minkowski

dataset = pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")

#A2-Label Encoding and One Hot Encoding
def label_encode(column):
    encoding={}
    encoded_column=[]
    current_label=0
    for value in column:
        if value not in encoding:
            encoding[value]=current_label
            current_label+=1
        encoded_column.append(encoding[value])
    return encoded_column,encoding
def one_hot_encode(column):
    unique_values=list(dict.fromkeys(column))
    one_hot_dictionary={}
    for category in unique_values:
        encoded_column=[]
        for value in column:
            if value==category:
                encoded_column.append(1)
            else:
                encoded_column.append(0)
        one_hot_dictionary[category]=encoded_column
    return one_hot_dictionary

#A3-Apply Label and One Hot Encoding
def apply_label_encoding(dataset):
    encoded_dataset=dataset.copy()
    categorical_columns=["Education","Marital_Status"]
    encoding_details={}
    for column in categorical_columns:
        encoded_values, mapping=label_encode(dataset[column])
        encoded_dataset[column]=encoded_values
        encoding_details[column]=mapping
    return encoded_dataset, encoding_details
def apply_one_hot_encoding(dataset):
    encoded_dataset=dataset.copy()
    categorical_columns=["Education","Marital_Status"]
    for column in categorical_columns:
        encoded_dictionary=one_hot_encode(dataset[column])
        encoded_dataframe=pd.DataFrame(encoded_dictionary)
        encoded_dataframe.columns=[column + "_" + str(name) for name in encoded_dataframe.columns]
        encoded_dataset.drop(column,axis=1,inplace=True)
        encoded_dataset = pd.concat([encoded_dataset, encoded_dataframe],axis=1)
    return encoded_dataset
def feature_dimension(dataset):
    return dataset.shape

#A4-Generalized Minkowski Distance
def minkowski_distance(vector1,vector2,p):
    if len(vector1)!=len(vector2):
        raise ValueError("Vectors must have same length.")
    distance_sum=0
    for i in range(len(vector1)):
        distance_sum+=abs(vector1[i]-vector2[i])**p
    distance=distance_sum**(1/p)
    return distance

#A5-Minkowski Distance for p=1 to 10
def calculate_minkowski_for_range(vector1,vector2):
    p_values=[]
    distance_values=[]
    for p in range(1,11):
        distance=minkowski_distance(vector1, vector2, p)
        p_values.append(p)
        distance_values.append(distance)
    return p_values,distance_values

#A6-Scipy Minkowski Comparison
def compare_with_scipy(vector1,vector2):
    comparison=[]
    for p in range(1,11):
        own_distance=minkowski_distance(vector1,vector2,p)
        scipy_distance=minkowski(vector1,vector2,p)
        comparison.append([p,own_distance,scipy_distance])
    return comparison

#A7-Dot Product,Euclidean Norm
def dot_product(vector1,vector2):
    result=0
    for i in range(len(vector1)):
        result+=vector1[i]*vector2[i]
    return result
def euclidean_norm(vector):
    total=0
    for value in vector:
        total+=value**2
    return math.sqrt(total)

#A8-Mean,Variance,Standard Deviation
def calculate_mean(data):
    total=0
    for value in data:
        total+=value
    return total/len(data)
def calculate_variance(data):
    mean=calculate_mean(data)
    variance_sum=0
    for value in data:
        variance_sum+=(value-mean)**2
    variance=variance_sum/len(data)
    return variance
def calculate_standard_deviation(data):
    variance=calculate_variance(data)
    return math.sqrt(variance)
def dataset_statistics(dataset):
    numeric_dataset=dataset.select_dtypes(include=[np.number])
    mean_vector=[]
    variance_vector=[]
    std_vector=[]
    column_names=numeric_dataset.columns.tolist()
    for column in column_names:
        values=numeric_dataset[column].tolist()
        mean_vector.append(calculate_mean(values))
        variance_vector.append(calculate_variance(values))
        std_vector.append(calculate_standard_deviation(values))
    return column_names,mean_vector,variance_vector,std_vector

#A9-Numpy Comparison
def numpy_statistics(dataset):
    numeric_dataset = dataset.select_dtypes(include=[np.number])
    mean_vector = numeric_dataset.mean(axis=0)
    variance_vector = numeric_dataset.var(axis=0)
    std_vector = numeric_dataset.std(axis=0)
    return mean_vector, variance_vector, std_vector

#A10-Histogram
def plot_histogram(dataset, feature_name):
    feature = dataset[feature_name]
    mean = np.mean(feature)
    variance = np.var(feature)
    plt.figure(figsize=(8,5))
    plt.hist(feature,bins=10,edgecolor="black")
    plt.title("Histogram of " + feature_name)
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    return mean, variance

#A11-Euclidean Distance,Assign Cluster,Update Centroids,KMEANS
def euclidean_distance(point1,point2):
    distance=0
    for i in range(len(point1)):
        distance+=(point1[i]-point2[i])**2
    return math.sqrt(distance)
def assign_clusters(data,centroids):
    clusters=[]
    for point in data:
        distances=[]
        for centroid in centroids:
            distances.append(euclidean_distance(point,centroid))
        cluster=distances.index(min(distances))
        clusters.append(cluster)
    return clusters
def update_centroids(data,clusters,k):
    new_centroids=[]
    data=np.array(data)
    for cluster in range(k):
        cluster_points=data[np.array(clusters)==cluster]
        if len(cluster_points)==0:
            new_centroids.append(data[0])
        else:
            new_centroids.append(np.mean(cluster_points,axis=0))
    return new_centroids
def kmeans(data,k,max_iterations=100):
    data=np.array(data)
    centroids=data[:k]
    for iteration in range(max_iterations):
        clusters=assign_clusters(data,centroids)
        new_centroids=update_centroids(data,clusters,k)
        if np.allclose(centroids,new_centroids):
            break
        centroids=new_centroids
    return clusters, centroids


#Main
print("A2:LABEL ENCODING")
education_encoded,education_mapping=label_encode(dataset["Education"])
marital_encoded, marital_mapping=label_encode(dataset["Marital_Status"])
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
education_onehot=one_hot_encode(dataset["Education"])
marital_onehot=one_hot_encode(dataset["Marital_Status"])
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
label_dataset,mappings=apply_label_encoding(dataset)
onehot_dataset=apply_one_hot_encoding(dataset)
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
numeric_dataset=label_dataset.select_dtypes(include=[np.number])
vector1=numeric_dataset.iloc[120].values
vector2=numeric_dataset.iloc[134].values
p=int(input("Enter value of p : "))
distance=minkowski_distance(vector1,vector2,p)
print("\nMinkowski Distance =", distance)

print("\n")
print("A5:MINKOWSKI DISTANCE (p = 1 TO 10)")
p_values,distance_values=calculate_minkowski_for_range(vector1,vector2)
for i in range(len(p_values)):
    print("p=",p_values[i],"Distance=",distance_values[i])
plt.figure(figsize=(8,5))
plt.plot(p_values,distance_values,marker="o")
plt.title("Minkowski Distance vs p")
plt.xlabel("p")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

print("\n")
print("A6:COMPARISON WITH SCIPY")
comparison=compare_with_scipy(vector1,vector2)
print("\nP\tOwn Function\tSciPy")
for row in comparison:
    print(row[0],"\t",round(row[1],6),"\t",round(row[2],6))

print("\n")
print("A7:DOT PRODUCT AND EUCLIDEAN NORM")
own_dot=dot_product(vector1,vector2)
numpy_dot=np.dot(vector1,vector2)
print("\nOwn Dot Product")
print(own_dot)
print("\nNumPy Dot Product")
print(numpy_dot)
own_norm_vector1=euclidean_norm(vector1)
numpy_norm_vector1=np.linalg.norm(vector1)
print("\nOwn Euclidean Norm")
print(own_norm_vector1)
print("\nNumPy Euclidean Norm")
print(numpy_norm_vector1)

print("\n")
print("A8:MEAN,VARIANCE,STANDARD DEVIATION")
column_names,mean_vector,variance_vector,std_vector=dataset_statistics(label_dataset)
print("\nFeature\t\tMean\t\tVariance\t\tStandard Deviation")
for i in range(len(column_names)):
    print(column_names[i],"\t",round(mean_vector[i],3),"\t",round(variance_vector[i],3),"\t",round(std_vector[i],3))

print("\n")
print("A9:COMPARISON WITH NUMPY")
numpy_mean,numpy_variance,numpy_std=numpy_statistics(label_dataset)
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
numeric_columns=label_dataset.select_dtypes(include=[np.number]).columns
print(list(numeric_columns))
feature=input("\nEnter Feature Name : ")
mean,variance=plot_histogram(label_dataset,feature)
print("\nMean =",mean)
print("Variance =",variance)

print("\n")
print("A11:K-MEANS CLUSTERING")
numeric_dataset=label_dataset.select_dtypes(include=[np.number])
data=numeric_dataset.values
k=int(input("\nEnter Number of Clusters : "))
clusters,centroids=kmeans(data,k)
print("\nFirst 40 Cluster Labels")
print(clusters[:40])
print("\nCentroids")
print(np.array(centroids))