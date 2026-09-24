import numpy as np
import matplotlib.pyplot as plt

# A1(a) SUMMATION UNIT
def summation_unit(x1, x2, w0, w1, w2):
    y = w0 + (w1 * x1) + (w2 * x2)
    return y

# A1(b) ACTIVATION UNIT - STEP FUNCTION
def step_activation(y):
    if y >= 0:
        return 1
    else:
        return 0

# A1(c) COMPARATOR UNIT
def comparator(target, output):
    error = target - output
    return error
def train_perceptron(inputs, targets, learning_rate,initial_weights, max_epochs=1000,convergence_error=0.002):
    weights = np.array(initial_weights, dtype=float)
    error_history = []
    for epoch in range(1, max_epochs + 1):
        sse = 0
        for i in range(len(inputs)):
            x1 = inputs[i][0]
            x2 = inputs[i][1]
            target = targets[i]
            y = summation_unit(x1, x2,weights[0],weights[1],weights[2])
            # Step activation
            output = step_activation(y)
            # Comparator
            error = comparator(target, output)
            # Weight update
            weights[0] = weights[0] + learning_rate * error
            weights[1] = weights[1] + learning_rate * error * x1
            weights[2] = weights[2] + learning_rate * error * x2
            # Sum Squared Error
            sse = sse + (error ** 2)
        error_history.append(sse)
        # Check convergence
        if sse <= convergence_error:
            return weights, epoch, error_history
    return weights, max_epochs, error_history

# A1 - BASIC DEMONSTRATION
def demonstrate_a1():
    x1 = 1
    x2 = 1
    w0 = 10
    w1 = 0.2
    w2 = -0.75
    target = 1
    y = summation_unit(x1, x2, w0, w1, w2)
    output = step_activation(y)
    error = comparator(target, output)
    print("\n================ A1 ================")
    print("Summation Unit Output:", y)
    print("Step Activation Output:", output)
    print("Comparator Error:", error)


# A2 - AND GATE
def run_a2_and_gate():
    # AND gate input
    inputs = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    # AND gate target
    targets = np.array([0, 0, 0, 1])
    # Given initial weights
    initial_weights = [10, 0.2, -0.75]
    # Given learning rate
    learning_rate = 0.05
    weights, epochs, error_history = train_perceptron(inputs,targets,learning_rate,initial_weights)
    print("\n================ A2 - AND GATE ================")
    print("Initial Weights: W0 = 10, W1 = 0.2, W2 = -0.75")
    print("Learning Rate:", learning_rate)
    print("Final Weights:", weights)
    print("Number of Epochs:", epochs)
    print("Final SSE:", error_history[-1])
    # Test final trained perceptron
    print("\nAND Gate Predictions:")
    for i in range(len(inputs)):
        x1 = inputs[i][0]
        x2 = inputs[i][1]
        y = summation_unit(x1, x2,weights[0],weights[1],weights[2])
        output = step_activation(y)
        print("Input:", x1, x2,"Target:", targets[i],"Output:", output)
    # Plot Epoch vs SSE
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, epochs + 1),error_history,marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Sum Squared Error (SSE)")
    plt.title("A2 - AND Gate: Epoch vs SSE")
    plt.grid(True)
    plt.show()
    return weights, epochs, error_history


# A4 - VARYING LEARNING RATE
def run_a4_learning_rates():
    # AND gate input
    inputs = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    # AND gate target
    targets = np.array([0, 0, 0, 1])
    # Initial weights must remain the same
    initial_weights = [10, 0.2, -0.75]
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0]
    epochs_required = []
    print("\n================ A4 - LEARNING RATE ================")
    print("\nLearning Rate\tEpochs")
    for learning_rate in learning_rates:
        weights, epochs, error_history = train_perceptron(inputs,targets,learning_rate,initial_weights)
        epochs_required.append(epochs)
        print(f"{learning_rate:.1f}\t\t{epochs}")
    # Plot Learning Rate vs Epochs
    plt.figure(figsize=(7, 5))
    plt.plot(learning_rates,epochs_required,marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Number of Epochs")
    plt.title("A4 - Learning Rate vs Number of Epochs")
    plt.grid(True)
    plt.show()
    return learning_rates, epochs_required

# A5 - XOR GATE
def run_a5_xor_gate():
    # XOR gate input
    inputs = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    # XOR gate target
    targets = np.array([0, 1, 1, 0])
    # Same initial weights as A2
    initial_weights = [10, 0.2, -0.75]
    # Same learning rate as A2
    learning_rate = 0.05
    weights, epochs, error_history = train_perceptron(inputs,targets,learning_rate,initial_weights)
    print("\n================ A5 - XOR GATE ================")
    print("Initial Weights: W0 = 10, W1 = 0.2, W2 = -0.75")
    print("Learning Rate:", learning_rate)
    print("Final Weights:", weights)
    print("Number of Epochs:", epochs)
    print("Final SSE:", error_history[-1])
    print("\nXOR Gate Predictions:")
    for i in range(len(inputs)):
        x1 = inputs[i][0]
        x2 = inputs[i][1]
        y = summation_unit(x1, x2,weights[0],weights[1],weights[2])
        output = step_activation(y)
        print("Input:", x1, x2,"Target:", targets[i],"Output:", output)
    # Plot Epoch vs SSE
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, epochs + 1),error_history)
    plt.xlabel("Epoch")
    plt.ylabel("Sum Squared Error (SSE)")
    plt.title("A5 - XOR Gate: Epoch vs SSE")
    plt.grid(True)
    plt.show()
    return weights, epochs, error_history

# MAIN PROGRAM
if __name__ == "__main__":
    demonstrate_a1()
    run_a2_and_gate()
    run_a4_learning_rates()
    run_a5_xor_gate()