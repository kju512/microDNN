import numpy as np
from net.net import neuralnetwork
from net.convLayer import ConvolutionLayer
from net.fcLayer import FcLayer
from net.poolingLayer import PoolingLayer
from activation.activation import acti_func
from dataproccess.data import loadmnistImages,loadmnistLabels
from optimizer.optimizer import arguments,optFunction

def mnist_LeNetdemo():
    #define activation function
    acti_f1=acti_func('tanh')
    acti_f2=acti_func('sigmoid')

    #define network structure
    net=neuralnetwork()
    conv1=ConvolutionLayer(32,32,5,1,6,acti_f1)
    pool2=PoolingLayer(28,28,2,6,'max',acti_f1)
    conv3=ConvolutionLayer(14,14,5,6,16,acti_f1)
    pool4=PoolingLayer(10,10,2,16,'max',acti_f1)
    conv5=ConvolutionLayer(5,5,5,16,120,acti_f1)
    fc6=FcLayer(120,10,acti_f2)

    net.addlayer(conv1)
    net.addlayer(pool2)
    net.addlayer(conv3)
    net.addlayer(pool4)
    net.addlayer(conv5)
    net.addlayer(fc6)
    

    #set training parameters
    min_batch_size=64
    epoch_num=1
    arg=arguments(0.9,0.0003,1e-7)
    opt=optFunction('momentum',arg)

    #load mnint dataset
    print('load data..............')
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
    print('start trainning..............')
    net.fit(data_x[:],data_y[:],min_batch_size,epoch_num,opt)
    print('end trainning..............')
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
    mnist_LeNetdemo()
