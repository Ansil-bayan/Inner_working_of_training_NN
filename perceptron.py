import numpy as np
import pandas as pd

rg = np.random.default_rng() #here the random number generator is assigned to rg
bias = 0.5

def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values)) #rg.random(rows,columns) gives a numpy array with rows rows and columns columns
    weights = rg.random((1, n_values))[0]
    targets = np.random.choice([0,1], n_features)
    '''
    data = pd.DataFrame(features, columns=["x0", "x1", "x2"])
    data["targets"] = targets
    
    print(f"{data} \nweights= {weights}")
    '''
    return features, weights, targets

# Activation Function
def sigmoid(w_sum):
    return 1/(1+np.e**(-w_sum))

def weighted_sum(features, weights, bias):
    return np.dot(features, weights) + bias

# Loss Function
def cross_entropy_loss(target, pred):
    return -(target*np.log10(pred)+(1-target)*(np.log10(1-pred)))

# Update Weights
def gradient_descent_correction(x, y, weights, bias, prediction):
    new_weights = []
    bias += learnrate*(y-prediction)

    for w,xi in zip(weights,x):
        new_weight = w + learnrate*(y-prediction)*xi
        new_weights.append(new_weight) 
    return new_weights, bias

# Data
'''
epochs = 10
learnrate = 0.1
    
errors = []

features = np.array(([0.1,0.5,0.2],[0.2,0.3,0.1],[0.7,0.4,0.2],[0.1,0.4,0.3]))
targets = np.array([0,1,0,1])
weights = np.array([0.4, 0.2, 0.6])
'''

def train(weights,bias):
    for i in range(epochs):
        for x, y in zip(features, targets): # x = feature[i] and y = targets[i]
            prediction = sigmoid(weighted_sum(x, weights, bias)) #does x.weights and adds bias and takes sigmoid of that number and returns another number
            weights, bias = gradient_descent_correction(x, y, weights, bias, prediction) #returns a list called new weights and a new bias
    
        print(f"weights: {weights} and bias: {bias}")

    # Printing out the log loss error on the training set
        out = sigmoid(weighted_sum(features, weights, bias)) #prediction with the new features and new bias
        loss = np.mean(cross_entropy_loss(targets, out)) 
        errors.append(round(float(loss),2))
       
a = generate_data(4,3)
features = a[0]
weights = a[1]
targets = a[2]
print(features,weights,targets)
epochs = 50
learnrate = 0.1
    
errors = []

train(weights,bias)

print(f"Average loss: {errors}") 


