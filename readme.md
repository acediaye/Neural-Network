# Machine Learning
Machine learning are programs that can discover their own algorithm instead of using preprogramed alogithms by the user. Artificial intelligence is a general term where the machine uses computer science and data to solve problems. Machine learning, a subcategory of artificial intelligence, is a machine that can learn and adapt automatically from experience without being explicitly programmed. Deep learning, a subcategory of machine learning, uses layers of algorithms to analyze data.

![image](pics/AI_hierarchy.png)

Machine learning have 3 main categories:
* Supervised learning: machine is presented with example inputs and their desired outputs and the goal is to learn a general rule that maps inputs to outputs.
* Unsupervised learning: No labels are given to the learning algorithm, the machine is to learn a general rule by itself.
* Reinforcement learning: A program interacts with a dynamic environment and perform certain goals (like a board game)

# Neural Network
Neural network is modeled after neurons by using many nodes in one layer connected to the next layer of nodes. Each node is connected to each node from the previous layer. Each connection line between nodes uses a weight/gain to show how much a node effects the next one. Each node is a summation of all the previous nodes times the associated weights. Then a function (activation) is applied to the result to rescale the large value to a range 0 to 1.

![image](pics/Neural_network_example.png)

A neural network calculates using forward propagation by inputing data on the input layer and using the associated weights to signify how much that specific node affects the next one. Then using a function to rescale all the values in the middle layers. The same process is repeated at each layer until the output layer. The output layer is the final calculation/answer. Then the neural network learns using backward propagation by comparing the calculated results with the desired results. The neural network is told what the answers should be and it needs to change the weights to cause the calculated results to match to desired results as close as possible. It changes to weights by the derivatives of the errors with respect to each weight.

# Forward Propagation
![image](pics/forward.PNG)

$$ y_1 = x_1 w_{11} + x_2 w_{12} + ... + x_i w_{1i} + b_1 $$

$$ y_2 = x_1 w_{21} + x_2 w_{22} + ... + x_i w_{2i} + b_2 $$

$$ y_3 = x_1 w_{31} + x_2 w_{32} + ... + x_i w_{3i} + b_3 $$

$$ y_j = x_1 w_{j1} + x_2 w_{j2} + ... + x_i w_{ji} + b_j $$

$$ 
\begin{bmatrix} y_1 \\\ y_2 \\\ ... \\\ y_j \end{bmatrix}_{jx1} =
\begin{bmatrix} w_{11} && w_{12} && ... && w_{1i} \\\ w_{21} && w_{22} && ... && w_{2i} \\\ ... && ... && && ... \\\ w_{j1} && w_{j2} && ... && w_{ji} \end{bmatrix}_{jxi}
\begin{bmatrix} x_1 \\\ x_2 \\\ ... \\\ x_i \end{bmatrix}_{ix1} +
\begin{bmatrix} b_1 \\\ b_2 \\\ ... \\\ b_j \end{bmatrix}_{jx1} 
$$

$$ Y = W * X + B $$

# Backward Propagation
$\frac{dE}{dY}$ derivative of error with respect to output

$\frac{dE}{dW}$ derivative of error with repect to weight

$\frac{dE}{dB}$ derivative of error with repect to bias

$\frac{dE}{dX}$ derivative of error with repect to input

$$
\frac{dE}{dY} = 
\begin{bmatrix}
\frac{dE}{dy_1} \\ \frac{dE}{dy_2} \\ ... \\ \frac{dE}{dy_j} 
\end{bmatrix}_{jx1}
$$

$$
\frac{dE}{dW} = 
\begin{bmatrix}
\frac{dE}{dw_{11}} && \frac{dE}{dw_{12}} && ... && \frac{dE}{dw_{1i}} \\\ \frac{dE}{dw_{21}} && \frac{dE}{dw_{22}} && ... && \frac{dE}{dw_{2i}} \\\ ... && ...&& && ... \\\ \frac{dE}{dw_{j1}} && \frac{dE}{dw_{j2}} && ... && \frac{dE}{dw_{ji}} 
\end{bmatrix}_{jxi}
$$

$$
\frac{dE}{dB} =
\begin{bmatrix}
\frac{dE}{db_1} \\\ \frac{dE}{db_2} \\\ ... \\\ \frac{dE}{db_j}
\end{bmatrix}_{jx1}
$$

$$
\frac{dE}{dX} = 
\begin{bmatrix}
\frac{dE}{dx_1} \\\ \frac{dE}{dx_2} \\\ ... \\\ \frac{dE}{dx_i}
\end{bmatrix}_{ix1}
$$

---
$$\frac{dE}{dw_{12}} = \frac{dE}{dy_1}\frac{dy_1}{dw_{12}} + \frac{dE}{dy_2}\frac{dy_2}{dw_{12}} + ... + \frac{dE}{dy_i}\frac{dy_i}{dw_{12}}$$

$$
\frac{dE}{dw_{12}} = \frac{dE}{dy_1} x_2 + \frac{dE}{dy_2} 0 + ... + \frac{dE}{dy_j} 0
$$

$$
\frac{dE}{dw_{ji}} = \frac{dE}{dy_j} x_i
$$

$$
\frac{dE}{dW} = 
\begin{bmatrix}
\frac{dE}{dy_1} x_1 && \frac{dE}{dy_1} x_2 && ... && \frac{dE}{dy_1} x_i \\\ \frac{dE}{dy_2} x_1 && \frac{dE}{dy_2} x_2 && ... && \frac{dE}{dy_2} x_i \\\ ... && ... && && ... \\\ \frac{dE}{dy_j} x_1 && \frac{dE}{dy_j} x_2 && ... && \frac{dE}{dy_j} x_i
\end{bmatrix}_{jxi}
$$

$$
\frac{dE}{dW} =
\begin{bmatrix}
\frac{dE}{dy_1} \\\ \frac{dE}{dy_2} \\\ ... \\\ \frac{dE}{dy_j}
\end{bmatrix}_{jx1}
\begin{bmatrix}
x_1 && x_2 && ... && x_i
\end{bmatrix}_{1xi}
$$

$$
\frac{dE}{dW} = 
\frac{dE}{dY} * X^T
$$

---
$$
\frac{dE}{db_1} =
\frac{dE}{dy_1}\frac{dy_1}{db_1} + \frac{dE}{dy_2}\frac{dy_2}{db_1} + ... + \frac{dE}{dy_j}\frac{dy_j}{db_1}
$$

$$
\frac{dE}{db_1} = 
\frac{dE}{dy_1} 1 + \frac{dE}{dy_2} 0 + ... + \frac{dE}{dy_j} 0
$$

$$
\frac{dE}{db_j} = \frac{dE}{dy_j}
$$

$$
\frac{dE}{dB} = \frac{dE}{dY}
$$

---
$$
\frac{dE}{dx_1} = 
\frac{dE}{dy_1}\frac{dy_1}{dx_1} + \frac{dE}{dy_2}\frac{dy_2}{dx_1} + ... + \frac{dE}{dy_j}\frac{dy_j}{dx_1}
$$

$$
\frac{dE}{dx_1} = \frac{dE}{dy_1} w_{11} + \frac{dE}{dy_2} w_{21} + ... + \frac{dE}{dy_j} w_{j1}
$$

$$
\frac{dE}{dx_i} = \frac{dE}{dy_1} w_{1i} + \frac{dE}{dy_2} w_{2i} + ... + \frac{dE}{dy_j} w_{ji}
$$

$$
\frac{dE}{dX} =
\begin{bmatrix}
\frac{dE}{dx_1} \\\ \frac{dE}{dx_x} \\\ ... \\\ \frac{dE}{dx_i}
\end{bmatrix} =
\begin{bmatrix}
\frac{dE}{dy_1} w_{11} + \frac{dE}{dy_2} w_{21} + ... + \frac{dE}{dy_j} w_{j1} \\\ \frac{dE}{dy_1} w_{12} + \frac{dE}{dy_2} w_{22} + ... + \frac{dE}{dy_j} w_{j2} \\\ ... \\\ \frac{dE}{dy_1} w_{1i} + \frac{dE}{dy_2} w_{2i} + ... + \frac{dE}{dy_j} w_{ji}
\end{bmatrix}_{ix1}
$$

$$
\frac{dE}{dX} = 
\begin{bmatrix}
w_{11} && w_{21} && ... && w_{j1} \\\ w_{12} && w_{22} && ... && w_{j2} \\\ ... && ... && && ... \\\ w_{1i} && w_{2i} && ... && w_{ji}
\end{bmatrix}_{ixj}
\begin{bmatrix}
\frac{dE}{dy_1} \\\ \frac{dE}{dy_2} \\\ ... \\\ \frac{dE}{dy_j}
\end{bmatrix}_{jx1}
$$

$$
\frac{dE}{dX} = w^T * \frac{dE}{dY}
$$

---
Backward propagation

$$
\frac{dE}{dW} = \frac{dE}{dY} * X^T
$$

$$
\frac{dE}{dX} = w^T * \frac{dE}{dY}
$$

$$
weights = weights - gain * \frac{dE}{dW}
$$

$$
bias = bias - gain * \frac{dE}{dY}
$$

# Activation Function: Sigmoid Function
Forward propagation

$$
y_1 = f(x_1)
$$

$$
y_2 = f(x_2)
$$

$$
y_i = f(x_i)
$$

$$
Y = f(X)
$$

Backward propagation

$$
\frac{dE}{dx_1} =
\frac{dE}{dy_1}\frac{dy_1}{dx_1}
$$

$$
\frac{dE}{dx_1} = \frac{dE}{dy_1} f'(x_1)
$$

$$
\frac{dE}{dx_i} = \frac{dE}{dy_i} f'(x_i)
$$

$$
\frac{dE}{dX} = \frac{dE}{dY} * f'(X)
$$

---
Sigmoid

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

$$
\frac{df}{dx} = (1+e^{-x})^{-1} =
-1(1 + e^{-x})^{-2} * (0 + e^{-x}) * -1
$$

$$
\frac{df}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{(1+e^{-x})} * \frac{e^{-x}}{(1 + e^{-x})}
$$

$$
\frac{df}{dx} = 
\frac{1}{(1+e^{-x})} * \frac{1 + e^{-x} - 1}{(1+e^{-x})} = \frac{1}{(1+e^{-x})} (\frac{1+e^{-x}}{(1+e^{-x})} - \frac{1}{(1+e^{-x})})
$$

$$
f'(x) = f(x) * (1 - f(x))
$$

![image](pics/sigmoid.png)

# Mean Square Error
Forward propagation

$$
E = \frac{1}{n}\sum(y_i^* - y_i)^2
$$

* $y^*$: desired output
* $y$: calculated output

Backward propagation

$$
\frac{dE}{dy_1} = 
\frac{d}{dy_1}\frac{1}{n}[(y_1^*-y_1)^2 + (y_2^*-y_2)^2 + ... + (y_n^*-y_n)^2]
$$ 

$$
\frac{dE}{dy_1} = 
\frac{1}{n}[2(y_1^*-y_1)^2*-1 + 0 + ... + 0]
$$ 

$$
\frac{dE}{dy_1} = 
\frac{2}{n}(y_1-y_1^*)
$$

$$
\frac{dE}{dy_i} = 
\frac{2}{n}(y_i-y_i^*)
$$

$$
\frac{dE}{dY} = \frac{2}{n}(Y-Y^*)
$$

# MNIST Database
Modified National Institude of Standards and Technology database is a large database of handwritten digits that is being used for training my machine learning model. Each digit 0-9 have 1000 grayscale pictures of 28 by 28 pixels. Each picture is fed into the model along with their labels for supervised learning. This model reaches ~85% correct (need fixing). The output is a vertical vector where each index represents a digit and the percent of certainty for the digit.

# Plots
![image](plots/0_pic.PNG)
```
answer: [0]
output:
 [[0.64642021]
 [0.00554115]
 [0.20715547]
 [0.00102692]
 [0.07240103]
 [0.00483149]
 [0.10223756]
 [0.00061372]
 [0.02285022]
 [0.00008676]]
predict: 0
```

![image](plots/1_pic.PNG)
```
answer: [1]
output:
 [[0.00217802]
 [0.61488584]
 [0.00103818]
 [0.00408102]
 [0.00110441]
 [0.07548588]
 [0.02037379]
 [0.01490281]
 [0.22501363]
 [0.000393  ]]
predict: 1
```

![image](plots/2_pic.PNG)
```
answer: [2]
output:
 [[0.01396745]
 [0.00342022]
 [0.86258393]
 [0.00456802]
 [0.00135811]
 [0.00064475]
 [0.00015516]
 [0.07318309]
 [0.00365399]
 [0.00021522]]
predict: 2
```

![image](plots/3_pic.PNG)
```
answer: [3]
output:
 [[0.00212244]
 [0.00000833]
 [0.01368471]
 [0.59453799]
 [0.01936649]
 [0.08031683]
 [0.00664245]
 [0.0409351 ]
 [0.05305306]
 [0.00059895]]
predict: 3
```

![image](plots/4_pic.PNG)
```
answer: [4]
output:
 [[0.00106467]
 [0.10811808]
 [0.000143  ]
 [0.02504691]
 [0.26558747]
 [0.02284418]
 [0.00360165]
 [0.05778329]
 [0.2156202 ]
 [0.00010401]]
predict: 4
```

![image](plots/5_pic.PNG)
```
answer: [5]
output:
 [[0.29984908]
 [0.28977966]
 [0.00529146]
 [0.04012997]
 [0.00120962]
 [0.60597091]
 [0.00519803]
 [0.00203414]
 [0.05669133]
 [0.00041489]]
predict: 5
```

![image](plots/6_pic.PNG)
```
answer: [6]
output:
 [[0.00012135]
 [0.24657283]
 [0.0078469 ]
 [0.00280392]
 [0.17059822]
 [0.00670571]
 [0.67092966]
 [0.00062509]
 [0.0420179 ]
 [0.00002527]]
predict: 6
```

![image](plots/7_pic.PNG)
```
answer: [7]
output:
 [[0.00000895]
 [0.07721025]
 [0.13787439]
 [0.00014047]
 [0.03250334]
 [0.00036179]
 [0.00047298]
 [0.70017081]
 [0.02929795]
 [0.00021321]]
predict: 7
```

![image](plots/8_pic.PNG)
```
answer: [8]
output:
 [[0.04246029]
 [0.00915137]
 [0.0145309 ]
 [0.04363076]
 [0.0732436 ]
 [0.1748903 ]
 [0.18932847]
 [0.00113845]
 [0.33205404]
 [0.00029844]]
predict: 8
```

![image](plots/9_pic.PNG)
```
answer: [9]
output:
 [[0.02861966]
 [0.00502164]
 [0.01220377]
 [0.00686151]
 [0.06386985]
 [0.03620384]
 [0.02198422]
 [0.16301425]
 [0.07836491]
 [0.62043225]]
predict: 9
```

# References
[Independent Code](https://www.youtube.com/watch?v=pauPCy_s0Ok)
