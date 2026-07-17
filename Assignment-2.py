import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#A1 
#Load Purchase Data
def load_purchase_data():
    purchase_data=pd.read_excel("Lab Session Data.xlsx",sheet_name="Purchase data")
    return purchase_data

#Separate Feature Matrix X and Output Vector y
def get_feature_output(purchase_data):
    X=purchase_data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
    y=purchase_data["Payment (Rs)"]
    return X,y

#Calculate Rank of Feature Matrix
def calculate_rank(X):
    feature_matrix=X.to_numpy()
    rank=np.linalg.matrix_rank(feature_matrix)
    return rank

#Calculate Product Cost using Pseudo Inverse
def calculate_product_cost(X, y):
    feature_matrix=X.to_numpy()
    output_vector=y.to_numpy()
    pseudo_inverse=np.linalg.pinv(feature_matrix)
    product_cost=pseudo_inverse @ output_vector
    return product_cost

#A3 
#Load Stock Data
def load_stock_data():
    stock_data=pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")
    return stock_data

#Calculate Mean 
def calculate_mean(price):
    total=0
    for value in price:
        total=total+value
    mean=total/len(price)
    return mean

#Calculate Variance
def calculate_variance(price):
    mean=calculate_mean(price)
    total=0
    for value in price:
        difference=value-mean
        total=total+(difference**2)
    variance=total/len(price)
    return variance

#Calculate Mean and Variance using NumPy
def numpy_statistics(price):
    mean=np.mean(price)
    variance=np.var(price)
    return mean,variance

# Compare Execution Time
def compare_execution_time(price):
    numpy_mean_time=0
    my_mean_time=0
    numpy_variance_time=0
    my_variance_time=0
    #NumPy Mean
    for i in range(10):
        start=time.perf_counter()
        np.mean(price)
        end=time.perf_counter()
        numpy_mean_time+=(end - start)
    # My Mean
    for i in range(10):
        start=time.perf_counter()
        calculate_mean(price)
        end=time.perf_counter()
        my_mean_time+=(end - start)
    # NumPy Variance
    for i in range(10):
        start=time.perf_counter()
        np.var(price)
        end=time.perf_counter()
        numpy_variance_time+=(end - start)
    # My Variance
    for i in range(10):
        start=time.perf_counter()
        calculate_variance(price)
        end=time.perf_counter()
        my_variance_time+=(end - start)
    average_numpy_mean=numpy_mean_time / 10
    average_my_mean=my_mean_time / 10
    average_numpy_variance=numpy_variance_time / 10
    average_my_variance=my_variance_time / 10
    return (average_numpy_mean,average_my_mean,average_numpy_variance,average_my_variance)

#Wednesday Mean
def calculate_wednesday_mean(stock_data):
    wednesday_data=stock_data[stock_data["Day"]=="Wed"]
    wednesday_price=wednesday_data["Price"]
    wednesday_mean=np.mean(wednesday_price)
    return wednesday_mean
#April Mean
def calculate_april_mean(stock_data):
    april_data=stock_data[stock_data["Month"]=="Apr"]
    april_price=april_data["Price"]
    april_mean=np.mean(april_price)
    return april_mean
#Probability of Loss
def probability_of_loss(stock_data):
    change=stock_data["Chg%"]
    loss_days=list(filter(lambda value: value < 0, change))
    probability=len(loss_days) / len(change)
    return probability
#Probability of Profit on Wednesday
def probability_profit_wednesday(stock_data):
    wednesday_profit=stock_data[(stock_data["Day"]=="Wed")&(stock_data["Chg%"] > 0)]
    probability=len(wednesday_profit) / len(stock_data)
    return probability
#Conditional Probability of Profit given Wednesday
def conditional_probability(stock_data):
    wednesday_data=stock_data[stock_data["Day"]=="Wed"]
    profit_days=wednesday_data[wednesday_data["Chg%"] > 0]
    probability=len(profit_days)/len(wednesday_data)
    return probability
#Scatter Plot
def plot_scatter(stock_data):
    plt.scatter(stock_data["Day"], stock_data["Chg%"])
    plt.xlabel("Day of the Week")
    plt.ylabel("Change Percentage")
    plt.title("Scatter Plot of Chg% vs Day")
    plt.grid(True)
    plt.show()

def main():
    purchase_data=load_purchase_data()
    X,y=get_feature_output(purchase_data)
    rank=calculate_rank(X)
    product_cost=calculate_product_cost(X, y)
    print("\nFeature Matrix (X)\n")
    print(X)
    print("\nOutput Vector (y)\n")
    print(y)
    print("\nRank of Feature Matrix :", rank)
    print("\nEstimated Cost of Products")
    print("Candies :", product_cost[0])
    print("Mangoes :", product_cost[1])
    print("Milk Packets :", product_cost[2])
    stock_data = load_stock_data()
    price = stock_data["Price"]
    numpy_mean, numpy_variance = numpy_statistics(price)
    my_mean = calculate_mean(price)
    my_variance = calculate_variance(price)
    (average_numpy_mean,average_my_mean,average_numpy_variance,average_my_variance)=compare_execution_time(price)
    wednesday_mean=calculate_wednesday_mean(stock_data)
    april_mean=calculate_april_mean(stock_data)
    loss_probability=probability_of_loss(stock_data)
    profit_wednesday=probability_profit_wednesday(stock_data)
    conditional_profit=conditional_probability(stock_data)
    print("\nNumPy Mean :",numpy_mean)
    print("My Mean :",my_mean)
    print("\nNumPy Variance :",numpy_variance)
    print("My Variance :",my_variance)
    print("\nAverage NumPy Mean Time :",average_numpy_mean)
    print("Average My Mean Time :",average_my_mean)
    print("\nAverage NumPy Variance Time :",average_numpy_variance)
    print("Average My Variance Time :",average_my_variance)
    print("\nPopulation Mean :",numpy_mean)
    print("Wednesday Mean :",wednesday_mean)
    print("April Mean :",april_mean)
    print("\nProbability of Loss :",loss_probability)
    print("Probability of Profit on Wednesday :",profit_wednesday)
    print("Conditional Probability :",conditional_profit)
    plot_scatter(stock_data)
if __name__ == "__main__":
    main()