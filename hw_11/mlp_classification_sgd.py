#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics import log_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Hidden layer size")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=797, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [np.random.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               np.random.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]

    relu = lambda x: np.maximum(x,0)

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs times weights[0]).
        # The value of the output layer is computed as softmax(hidden_layer times weights[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        Z = np.dot(weights[0],inputs)
        A = relu(Z)
        output = np.dot(weights[1], A)
        return softmax(output,axis = 0), A
    
    for iteration in range(args.iterations):
        permutation = np.random.permutation(train_data.shape[0])
        permuted_x_train, permuted_y_train = (
            train_data[permutation],
            train_target[permutation],
        )
        batch_count = int(train_data.shape[0] / args.batch_size)

        for batch_x, batch_y in zip(
            np.split(permuted_x_train, batch_count),
            np.split(permuted_y_train, batch_count),
        ):
            probs,A = forward(batch_x)
            
            dZ2 = probs - batch_y
            dW2 = 1/m*np.dot(dZ2,A.T)
            dZ1 = np.dot(weights[1],dZ2)*(1-np.power(A,2))
            dW1 = 1/m*np.dot(dZ1,batch_x.T)
            
            weights[1] -= args.learning_rate*dW2
            weights[0] -= args.learning_rate*dW1

        train_probs,_ = forward(train_data)
        train_pred = np.argmax(train_probs)
        test_probs,_ = forward(test_data)
        train_pred = np.argmax(test_probs)
        
        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1,
            100 * sklearn.metrics.accuracy_score()# Training accuracy,
            100 * # Test accuracy,
        ))
