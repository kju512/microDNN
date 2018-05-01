#written by Michael chen

import numpy as np
from net.net import neuralnetwork
from net.fcLayer import FcLayer
from activation.activation import acti_func
from dataproccess.data import loadmnistImages,loadmnistLabels
from optimizer.optimizer import arguments,optFunction

def mnist_FcNetdemo():
    #define activation function
    acti_f=acti_func('sigmoid')

    #define network structure
    net=neuralnetwork()
    fc1=FcLayer(1024,25,acti_f)
    fc2=FcLayer(25,10,acti_f)
    net.addlayer(fc1)
    net.addlayer(fc2)

    #set training parameters
    min_batch_size=256
    epoch_num=30
    arg=arguments(0.9,0.03,1e-8)
    opt=optFunction('gradientdescent',arg)


    #load mnint dataset
    trainimages=loadmnistImages(r'/home/kju512/NewDisk/DeepLearning/deeplearningpubdata/mnist/train-images.idx3-ubyte')
    trainlabels=loadmnistLabels(r'/home/kju512/NewDisk/DeepLearning/deeplearningpubdata/mnist/train-labels.idx1-ubyte')
    testimages=loadmnistImages(r'/home/kju512/NewDisk/DeepLearning/deeplearningpubdata/mnist/t10k-images.idx3-ubyte')
    testlabels=loadmnistLabels(r'/home/kju512/NewDisk/DeepLearning/deeplearningpubdata/mnist/t10k-labels.idx1-ubyte')
    #set_trace
    data_y=np.zeros([trainlabels.size,10])
    data_x=np.zeros([trainimages.shape[0],1,trainimages.shape[1],trainimages.shape[2]])
    testdata_y=np.zeros([testlabels.size,10])
    testdata_x=np.zeros([testimages.shape[0],1,testimages.shape[1],testimages.shape[2]])


    #tranform lable data to one hot type    
    for i in range(trainlabels.size):
	    data_y[i][int(trainlabels[i,0])]=1
    for i in range(trainlabels.size):
	    data_x[i,0,:,:]=trainimages[i]
    for i in range(testlabels.size):
	    testdata_y[i][int(testlabels[i,0])]=1
    for i in range(testlabels.size):
	    testdata_x[i,0,:,:]=testimages[i]
    
    #debug
    #d=net.checkgrad(data_xxx[:50],data_yy[:50])
    #s=np.arange(d.shape[0])
    #np.random.shuffle(s)
    #print d[s[:100]]
    #print d[s[:]]
    #set_trace()

    #fit train dataset 
    net.fit(data_x[:],data_y[:],min_batch_size,epoch_num,opt)
    #calculate accuracy
    train_accuracy=net.compute_predict_rate(data_x[:],trainlabels[:])
    print("train accuracy is %f"%(train_accuracy))
    test_accuracy=net.compute_predict_rate(testdata_x[:],testlabels[:])
    print("test accuracy is %f"%(test_accuracy))
    
    #make a prediction
    predict_y=net.predict(testdata_x[:100])
    print('test samples first 100 results:')
    print(np.vstack((testlabels[:100].T,predict_y.T)).T)

if __name__=='__main__':
    mnist_FcNetdemo()
