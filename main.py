import math
import random

random.seed(20)

def rand(a, b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    return math.tanh(x)
    # return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return 1.0 - y**2
    # return y * (1 - y)

class NN:
    def __init__(self, ni, nh, no):
        self.ni = ni
        self.nh = nh
        self.no = no

        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        r1 = -2.0
        r2 = 2.0
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(r1, r2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(r1, r2)


    def update(self, inputs):

        for i in range(self.ni):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N):

        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change

        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error



        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))
        # print(self.update(patterns))


    def train(self, patterns, iterations=10000, N=rand(0.1,0.9)):
        print("Learning Rate",N)
        # j = 0.1
        # for i in range(10):
        #     j += 0.1
        #     print(j)
        for i in range(iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)

                self.backPropagate(targets, N)

def main():
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    n = NN(2, 2, 1)
    n.train(pat)
    # i = input("Enter Pattern:")
    # i = int(i)
    # j = input()
    # j = int(j)

    n.test(pat)



if __name__ == '__main__':
    main()
