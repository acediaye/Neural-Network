import numpy as np
import matplotlib.pyplot as plt

# Boolean table data
train = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
label = np.array([[0],
                  [1],
                  [1],
                  [0]])

# 2 -> 3 -> 1
# o -> o -> o
# init weights and bias uniform random
w_ih = np.random.randn(3, 2)
w_ho = np.random.randn(1, 3)
b_ih = np.zeros((3, 1))
b_ho = np.zeros((1, 1))

epochs = 10000
alpha = 0.1
errlist = np.zeros((epochs, 1))
# looping over entire dataset many times
for e in range(0, epochs):
    error = 0
    for i in range(0, len(train)):
        input1 = train[[i],:].T
        answer = label[i]
        # forward propagation
        hidden = w_ih@input1 + b_ih  # 3x1
        hidden = 1/(1+np.exp(-hidden))
        output = w_ho@hidden + b_ho  # 1x1
        output = 1/(1+np.exp(-output))
        # error
        error = error + 1/len(output)*(answer - output)**2  # 1x1
        errorprime = 2/len(output)*(output - answer)  # 1x1
        # backward propagation
        out2_grad = errorprime*output*(1-output)  # 1x1
        w2_grad = out2_grad@hidden.T  # 1x1 * 3x1' = 1x3
        in2_grad = w_ho.T@out2_grad  # 1x3' * 1x1 = 3x1
        w_ho = w_ho - alpha*w2_grad  # 1x3 - 1x3
        b_ho = b_ho - alpha*out2_grad  # 1x1 - 1x1
        
        out1_grad = in2_grad*hidden*(1-hidden)  # 3x1
        w1_grad = out1_grad@input1.T  # 3x1 * 2x1' = 3x2
        in1_grad = w_ih.T@out1_grad  # 3x2' * 3x1 = 2x1
        w_ih = w_ih - alpha*w1_grad  # 3x2 - 3x2
        b_ih = b_ih - alpha*out1_grad  # 3x1 - 3x1
    print(f'{e+1}/{epochs}')
    errlist[e] = error

# plot error
# plt.figure(1)
# time = np.arange(0, epochs)
# plt.plot(time, errlist)
# plt.show()

for i in range(0, len(train)):
    input1 = train[[i],:].T
    # forward
    hidden = w_ih@input1 + b_ih  # 3x1
    hidden = 1/(1+np.exp(-hidden))
    output = w_ho@hidden + b_ho  # 1x1
    output = 1/(1+np.exp(-output))
    print(f'input: {input1.T}, output: {output}')

print('done')

# https://www.youtube.com/watch?v=pauPCy_s0Ok
