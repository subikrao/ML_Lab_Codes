import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import math
'''
A1. DEVELOP A PERCEPTRON FOR AND GATE LOGIC
    Identify no. of epochs needed to reach converged learning
    Plot no. of epochs against error values after each epoch
'''
# making a dataframe to store the truth table of AND gate
# will use this df as training data
AND= { 
    'A': [0,0,1,1],
    'B': [0,1,0,1],
    'Z': [0,0,0,1]
}

AND_df = pd.DataFrame(AND)
print(AND_df)
A_B = AND_df.iloc[0:, 0:2]  # extracting A & B values
Z = AND_df.iloc[0:, 2]      # extracting Z values

#converting to numpy array
inputs=np.array(A_B)
outputs=np.array(Z)

#function to implement step function
#if x<0, y=0
#if x>=0, y=1
def step(sum):
    if(sum <= 0):
        return 0
    else:
        return 1

def perceptron(activation_function):
    #the given values for w0,w1,w2 and learning rate
    alpha=0.05
    W0 = 10
    W1 = 0.2
    W2 = -0.75

    epochs = range(1,1001) # no. of epochs varying from 1 to 1000
    ssq_errors = [] # to store the error value corresponding to no. of epochs
    point_of_convergence=0
    for i in epochs:
        sum_square_error = 0
        for j in range(0,4): #for each set of inputs A and B. (4 rows)
            a = inputs[j][0]
            b = inputs[j][1]
            # print("a = ", a)
            # print("b = ", b)
            summation = (W0)+(W1*a)+(W2*b)  # calculating sum according to perceptron diagram
            target = outputs[j]             
            if (activation_function(summation)==0):
                prediction = 0
            else:
                prediction = 1

            error_temp = target - prediction  # Ei = Ti - Pi
            # print("target=",target)
            # print("prediction=",prediction)
            sum_square_error += error_temp * error_temp  # square of error
            # print("sum_square_error=",sum_square_error)
            delta_W = -(alpha)*(target - prediction) # delta_W = -(alpha)(Ti-Pi)
            # print("delta_W=",delta_W)
            # delta_Wi = -(alpha)(Ti-Pi)(Xi)
            # Wi = Wi + delta_Wi
            W0 = W0 + (alpha*error_temp* 1)
            W1 = W1 + (alpha*error_temp* a)
            W2 = W2 + (alpha*error_temp* b)
            # print("W0=",W0)
            # print("W1=",W1)
            # print("W2=",W2,"\n")
        ssq_errors.append(sum_square_error) #store values of error

        if(sum_square_error <= 0.002):
            print("The learning has converged at epoch ", i+1)
            point_of_convergence=i
            break
    # print("Sum square errors till point of convergence: \n",ssq_errors)
    return (epochs, ssq_errors, point_of_convergence)

alpha_initial=0.05
W0_initial = 10
W1_initial = 0.2
W2_initial = -0.75
print("\nInitial weights: \nW0=",W0_initial)
print("W1=",W1_initial)
print("W2=",W2_initial)
print("learning rate=",alpha_initial)
step_epochs, step_ssq_errors, step_point_of_convergence = perceptron(step)
print("Sum square errors till point of convergence: \n",step_ssq_errors)

# Plotting the sum-square error values against no. of epochs
mp.figure(figsize=(20,6))
mp.plot(step_epochs[0:step_point_of_convergence] ,step_ssq_errors, marker='.') #let x axis show the no. of epochs and y axis shows the error values
mp.xticks(range(0,step_point_of_convergence+1,5))
mp.yticks([0,1,2,3,4,5,6])
mp.xlabel("Error values")
mp.ylabel("Number of epochs")
mp.title("Plot of error against epochs")
mp.grid(True)
mp.show()

# A2. Repeat the above A1 experiment with following activation functions. Compare the iterations
# taken to converge against each of the activation functions. Keep the learning rate same as A1.
# • Bi-Polar Step function
# • Sigmoid function
# • ReLU function

#BI-POLAR STEP FUNCTION
def bipolar_step(sum):
    if( sum > 0 ):
        return 1
    elif( sum == 0 ):
        return 0
    elif( sum < 0 ):
        return -1
print("\nInitial weights: \nW0=",W0_initial)
print("W1=",W1_initial)
print("W2=",W2_initial)
print("learning rate=",alpha_initial)    
bipolar_epochs, bipolar_ssq_errors, bipolar_point_of_convergence=perceptron(bipolar_step)
print("Sum square errors till point of convergence: \n",bipolar_ssq_errors)
#using same method for bi-polar step function as the activation function

#SIGMOID FUNCTION
def sigmoid(sum):
    result = 1/(1+math.exp(-sum))
    return result

print("\nInitial weights: \nW0=",W0_initial)
print("W1=",W1_initial)
print("W2=",W2_initial)
print("learning rate=",alpha_initial)  
sigmoid_epochs, sigmoid_ssq_errors, sigmoid_point_of_convergence=perceptron(sigmoid)
print("Sum square errors till point of convergence: \n",sigmoid_ssq_errors)
#using same method for sigmoid function as the activation function

#ReLU FUNCTION
def ReLU(sum):
    if(sum>0):
        return sum
    else:
        return 0
print("\nInitial weights: \nW0=",W0_initial)
print("W1=",W1_initial)
print("W2=",W2_initial)
print("learning rate=",alpha_initial)  
relu_epochs, relu_ssq_errors, relu_point_of_convergence=perceptron(ReLU)
print("Sum square errors till point of convergence: \n",relu_ssq_errors)
#using same method for bi-polar step function as the activation function

# A3. Repeat A1 with varying learning rates, keeping the initial weights as the same. 
# Learning rates = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}. 
# Plot the no. of iterations till convergence vs. the learning rate