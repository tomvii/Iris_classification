# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import division
import math
import random
import string
import pickle

flowerLables = {0:'Iris-setosa',
                 1:'Iris-versicolor',
                 2:'Iris-virginica'}
random.seed(0)
# Generate random number between a and b
def rand(a, b):
    return (b-a)*random.random() + a

# I * J size matrix (zero)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# sigmoid function
def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2

class NN:
    ''' Three-layer neural network '''
    def __init__(self, ni, nh, no):
        # Nodes number of input layer, hidden layer and output layer
        self.ni = ni + 1 # additional bias node
        self.nh = nh
        self.no = no

        # Activation
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # weight matrix
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # random init
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # momemtum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('Mismatch')

        # Activation input
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # Activation hidden layer
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # Activation output layer
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        # print self.ao
        return self.ao[:]

    def backPropagate(self, targets, N, M):
        ''' Backpropagation '''
        
        # distance between output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # hidden feature loss
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # Output layer weight update 
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print(N*change, M*self.co[j][k])

        # input weight update
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # Loss
        error = 0.0
        # for k in range(len(targets)):
        #     error = error + 0.5*(targets[k]-self.ao[k])**2
        error += 0.5*(targets[k]-self.ao[k])**2
        return error

    def test(self, patterns):
        count = 0
        for p in patterns:
            target = flowerLables[(p[1].index(1))]
            result = self.update(p[0])
            index = result.index(max(result))
            print(p[0], ': (Predict)', target, '-> (Ground Truth)', flowerLables[index])
            count += (target == flowerLables[index])
            # result_str = flowerLables[index]
            # if target == result_str:
            #     count += 1
            # else:
            #     pass
        accuracy = float(count/len(patterns))
        print('accuracy: %-.9f' % accuracy)




    def weights(self):
        print('Input layer weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output layer weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.1, M=0.01):
        # N: (learning rate)
        # M: (momentum factor)
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('Loss of {} iteration: {}'.format(i, error) )




import numpy as np
import pandas as pd

# features 0-3
# labels 4
def iris():
    data = []
    # read dataset
    raw = pd.read_csv('iris.csv')
    raw_data = raw.values
    raw_feature = raw_data[0:,0:4]
    for i in range(len(raw_feature)):
        ele = []
        ele.append(list(raw_feature[i]))
        if raw_data[i][4] == 'Iris-setosa':
           ele.append([1,0,0])
        elif raw_data[i][4] == 'Iris-versicolor':
            ele.append([0,1,0])
        else:
            ele.append([0,0,1])
        data.append(ele)

    # print data

    random.shuffle(data)
    # print data
    training = data[0:100]
    test = data[101:]
    # print np.shape(l)
    # print np.shape(data)
    # training_set = np.c_[data, l]
    nn = NN(4,7,3)
    nn.train(training,iterations=50)

    # # save weights
    # with open('wi.txt', 'w') as wif:
    #     pickle.dump(nn.wi, wif)
    # with open('wo.txt', 'w') as wof:
    #     pickle.dump(nn.wo, wof)

    nn.test(test)

if __name__ == '__main__':
    iris()

