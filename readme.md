# Micro DNN
### Introduction
The repository is a micro DNN framework written by Michael Chen.The purpose of the project is to comprehensively understand the mathematic principle of deep neuralwork.  
The project has implement fully connected layer,convolutional layer,pooling layer which are mainly used in DNN.  
We also give two demo examples which illustrate how to use it.Both of them are used for mnist dataset.They illustrate a classifier network respectively.  
One example is a fully connected nueral network which has 1024 input units and 25 hidden units and 10 output units.  
Another example is a convolutional network,it is a very very famous network called LeNet which can be found [here](http://yann.lecun.com/exdb/lenet/index.html)

### How to run the demos
* Clone this repository.
* Download mnist dataset( which can be found [here](http://yann.lecun.com/exdb/mnist/))to your own directory.
* Modify the code which load mnist dataset from the directory stored in the LeNetdemo.py and FCdemo.py
* run the code under the repository directory:  
`$python FCdemo.py`

### Dependency
* numpy-----the only package which is used in constructing the framework. 
* struct----it is used in loading mnist dataset


### Other questions
The project is wrriten under python3.  
Because it is a very simple dnn framework and no use GPU to accelerate calculations,it will cost a long time to run LeNetdemo.py ,it may be several hours to run it.  
the FCdemo.py will cost few time to run. it just need about twenty minutes to complete training on my computer.

### Wecome discussion
my email is chenzh8312@sina.com.cn
