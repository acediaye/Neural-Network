import numpy as np
import matplotlib.pyplot as plt

# create dataset and labels from images
path = r'C:\Program Files\MATLAB\R2018a\toolbox\nnet\nndemos\nndatasets\DigitDataset'
data = np.zeros((10000, 28*28), dtype=np.uint8)
labels = np.zeros((10000, 1), dtype=np.uint8)
k = 0  # fill data 10,000 rows
for i in range(0, 9):  # digits 0-9
    folder = f'\{i}'
    for j in range(1, 1000+1):  # images 1000
        if i == 0:
            filename = f'\image{9*1000+j}.png'
        elif i == 1:
            filename = f'\image{j}.png'
        else:
            filename = f'\image{(i-1)*1000+j}.png'
        file = path + folder + filename
        image = plt.imread(file)
        image = image*255
        image = image.astype(np.uint8)
        k = k + 1
        data[k,:] = np.reshape(image, (1,28*28))
        labels[k,:] = i
        
# shuffle rows in data and label 
temp = np.arange(10000)
np.random.shuffle(temp)
data = data[temp,:]
labels = labels[temp,:]

# split data into training set and test set
split = 0.9
train_data  = data[0:int(split*10000), :]
train_label = labels[0:int(split*10000), :]
test_data   = data[int(split*10000):, :]
test_label  = labels[int(split*10000):, :]

# init weights and bias uniform random
w_ih = np.random.randn(20, 28*28)
w_ho = np.random.randn(10, 20)
b_ih = np.zeros((20, 1))
b_ho = np.zeros((10, 1))
encoding = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
alpha = 0.1
epochs = 200

# train model
for j in range(0, epochs):
# for j in range(0, 1):  # epochs
    correct = 0
    for i in range(0, len(train_data)):
    # for i in range(0, 1):
        input1 = train_data[[i],:].astype(np.float64).T
        answer = np.roll(encoding, train_label[i,:]).T
        # forward propagation
        hidden = w_ih@input1 + b_ih  # 20x1
        hidden = 1/(1+np.exp(-hidden))
        output = w_ho@hidden + b_ho  # 10x1
        output = 1/(1+np.exp(-output))
        if np.argmax(output) == train_label[i,:]:
            correct = correct + 1
        # error
        error = 1/len(output)*(answer - output)**2  # 10x1
        errorprime = 2/len(output)*(output - answer)  # 10x1
        # backward propagation
        # output to hidden
        out2_grad = errorprime*output*(1-output)  # 10x1
        w2_grad = out2_grad@hidden.T  # 10x1 * 20x1' = 10x20
        in2_grad = w_ho.T@out2_grad  # 10x20' * 10x1 = 20x1
        w_ho = w_ho - alpha*w2_grad  # 10x20 - 10x20
        b_ho = b_ho - alpha*out2_grad  # 10x1 - 10x1
        # hidden to input
        out1_grad = in2_grad*hidden*(1-hidden)  # 20x1
        w1_grad = out1_grad@input1.T  # 20x1 * 784x1' = 20x784
        in1_grad = w_ih.T@out1_grad  # 20x784' * 20x1 = 784x1
        w_ih = w_ih - alpha*w1_grad  # 20x784 - 20x784
        b_ih = b_ih - alpha*out1_grad  # 20x1 - 20x1
    print(f'loop: {j+1}, correct: {correct/len(train_data)*100:.2f}%')

# run model
np.set_printoptions(suppress=True) # don't use scientific notation
loop = True
while(loop):
    x = input('enter index number:')
    if x.isnumeric():
        # print(x)
        input1 = test_data[[int(x)],:].astype(np.float64).T
        # forward propagation
        hidden = w_ih@input1 + b_ih
        hidden = 1/(1+np.exp(-hidden))  # value too large or too small
        output = w_ho@hidden + b_ho
        output = 1/(1+np.exp(-output))
        print(f'answer: {test_label[int(x),:]}')
        print(f'output:\n {output}')
        print(f'predict: {np.argmax(output)}')
        plt.figure(1)
        plt.imshow(np.reshape(test_data[[int(x)],:], (28, 28)), cmap='gray')
        plt.show()
    else:
        loop = False

print('done')

# https://www.youtube.com/watch?v=pauPCy_s0Ok

# path = r'C:\Program Files\MATLAB\R2018a\toolbox\nnet\nndemos\nndatasets\DigitDataset'
# file = r'\0\image9001.png'
# image = plt.imread(path+file)
# image = image*255
# image = image.astype(np.uint8)
# plt.figure(1)
# plt.imshow(image, cmap='gray')
# print(np.shape(image))
# print(type(image[0,0]))
# print(image[12,7])
# plt.show()
