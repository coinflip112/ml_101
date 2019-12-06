#!/usr/bin/env python3
# dcfac0e3-1ade-11e8-9de3-00505601122b
# dce9cf60-42b6-11e9-b0fd-00505601122b
# 7d179d73-3e93-11e9-b0fd-00505601122b
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel

def kernel(x,y,gamma,degree,kernel):
    
    if kernel == 'poly':
        xTy = x.T * y
        result = gamma * xTy + 1

        return result**degree
    elif kernel == 'rbf':
        euclid = np.sum( (x - y) ** 2 )
        result = -gamma * euclid

        return  np.exp(result)

def loss(true,preds,weights,bias):
    
    
    bias = 0
    result =  1/2 * np.square(true - preds - bias)
    result = np.mean(result) + 1/2 * args.l2 * np.dot(weights,weights)
    
    return result

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default=50, type=int, help="Number of examples")
    parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--iterations", default=1000, type=int, help="Number of training iterations")
    parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.examples)
    train_targets = np.sin(5 * train_data) + np.random.normal(scale=0.25, size=args.examples) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.examples)
    test_targets = np.sin(5 * test_data)+ 1
    

    coefs = np.zeros(args.examples)
    # TODO: Perform `iterations` of SGD-like updates, but in dual formulation
    # using `coefs` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is MSE with L2 regularization:
    #   L = sum_{i=1}^N [1/2 * (target_i - phi(x_i)^T w - bias)^2] + 1/2 * args.l2 * w^2
    #
    # For bias use explicitly the average of training targets, and do not update
    # it futher during training.
    #
    # Instead of using feature map `phi` directly, we use given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following kernels:
    # - poly: K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - rbf: K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each update print RMSE both on training and testing data.
    
    if args.kernel == 'rbf':
        kernel_matrix_train = rbf_kernel(train_data.reshape(-1,1), gamma = args.kernel_gamma) 
        kernel_matrix_test  = rbf_kernel(train_data.reshape(-1,1),test_data.reshape(-1,1),gamma = args.kernel_gamma)
    else:
        kernel_matrix_train = polynomial_kernel(train_data.reshape(-1,1), gamma = args.kernel_gamma, degree = args.kernel_degree) 
        kernel_matrix_test  = polynomial_kernel(train_data.reshape(-1,1),test_data.reshape(-1,1),
                              gamma = args.kernel_gamma,degree = args.kernel_degree)

    #for i in range(train_data.shape[0]):
        
    #    xi = train_data[i]

    #    for j in range(train_data.shape[0]):
            
    #        xj = train_data[j] 

    #        kernel_matrix_train[i,j] = kernel(x = xi, y = xj, 
    #                                         kernel = args.kernel, 
    #                                         gamma = args.kernel_gamma, 
    #                                         degree = args.kernel_degree)
    
    #for i in range(test_data.shape[0]):
        
    #    xi = test_data[i]

    #    for j in range(train_data.shape[0]):
            
    #        xj = train_data[j] 

    #        kernel_matrix_test[i,j] = kernel(x = xi, y = xj, 
    #                                        kernel = args.kernel, 
    #                                        gamma = args.kernel_gamma, 
    #                                        degree = args.kernel_degree)

    
    

    for iteration in range(args.iterations):
        # TODO
        
        permutation = np.random.permutation(train_data.shape[0])
        
        coefs += args.learning_rate * (train_targets - np.matmul(kernel_matrix_train,coefs) - np.mean(train_targets) -  args.l2 * coefs)
        
        train_preds = np.matmul(kernel_matrix_train,coefs) + np.mean(train_targets)
        test_preds = np.matmul(kernel_matrix_test.T,coefs) + np.mean(train_targets)
        
        l_train = np.sqrt(np.mean( np.square(train_targets - train_preds  )))
        l_test = np.sqrt(np.mean( np.square(test_targets - test_preds  )))


        print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
            iteration + 1,
            # RMSE on train data,
            l_train,
            # RMSE on test data,
            l_test
        ))
    

    if args.plot:
        test_predictions = np.matmul(kernel_matrix_test.T,coefs) + np.mean(train_targets)

        plt.plot(train_data, train_targets, "bo", label="Train targets")
        plt.plot(test_data, test_targets, "ro", label="Test targets")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend(loc="upper left")
        plt.show()
