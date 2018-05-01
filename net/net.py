#written by Michael chen

import numpy as np
from net.convLayer import ConvolutionLayer
from net.fcLayer import FcLayer
from net.poolingLayer import PoolingLayer
from loss.loss import MeanSquareErorrLoss,crossEntropyLoss
from activation.activation import acti_func

class neuralnetwork(object):
    def __init__(self):
        self.layers=[]
    def addlayer(self,newlayer):
        self.layers.append(newlayer)
    def forwardpropagation(self,data_x):
        layersnum=len(self.layers)
        datanum=data_x.shape[0]
        y=data_x
        for i in range(layersnum):
            if self.layers[i].name == 'conv':
                y=y.flatten().reshape(datanum,self.layers[i].inchannels,self.layers[i].inputH,self.layers[i].inputW)
            if self.layers[i].name == 'pool':
                y=y.flatten().reshape(datanum,self.layers[i].inchannels,self.layers[i].inputH,self.layers[i].inputW)
            if self.layers[i].name == 'fc':
                y=y.flatten().reshape(datanum,self.layers[i].inputsize)
            y=self.layers[i].forwardpropagation(y)
        return y
    def backwardpropagation(self,data_x,data_y):
        deltas=np.array([])
        y=self.forwardpropagation(data_x)
        datanum=data_x.shape[0]
        layersnum=len(self.layers)
        deltas=y-data_y
        for i in np.argsort(-np.arange(layersnum)):
            if self.layers[i].name == 'conv':
                deltas=deltas.flatten().reshape(datanum,self.layers[i].outchannels,self.layers[i].outputH,self.layers[i].outputW)
            if self.layers[i].name == 'pool':
                deltas=deltas.flatten().reshape(datanum,self.layers[i].outchannels,self.layers[i].outputH,self.layers[i].outputW)
            if self.layers[i].name == 'fc':
                deltas=deltas.flatten().reshape(datanum,self.layers[i].outputsize)
            if i!=0:
                self.layers[i].backwardpropagation(self.layers[i].inputdata,deltas,self.layers[i-1].acti)
            else:
                self.layers[i].backwardpropagation(self.layers[i].inputdata,deltas,acti_func('identity'))
            deltas=self.layers[i].inputdeltas
    def compute_loss(self,data_x,data_y):
        y=self.forwardpropagation(data_x)
        r=-np.sum(np.mean(np.log(y)*data_y+(1.0-data_y)*np.log(1.0-y),0))
        return r
    def checkgrad(self,data_x,data_y):
        eta=1e-8
        dw=np.array([])
        dw_=np.array([])
        layersnum=len(self.layers)
        self.backwardpropagation(data_x,data_y)
        for i in range(layersnum):
            #set_trace()
            if self.layers[i].name == 'fc':
                dw=np.hstack((dw,self.layers[i].dW.flatten()))
                dw=np.hstack((dw,self.layers[i].db.flatten()))
            if self.layers[i].name == 'conv':
                dw=np.hstack((dw,self.layers[i].dW.flatten()))
                dw=np.hstack((dw,self.layers[i].db.flatten()))
            #set_trace()
        for i in range(layersnum):
            if self.layers[i].name == 'fc': 
                for j in range(self.layers[i].W.shape[0]):
                    for v in range(self.layers[i].W.shape[1]):
                        ww=self.layers[i].W[j][v]
                        self.layers[i].W[j][v]=ww+eta
                        loss1=self.compute_loss(data_x,data_y)
                        self.layers[i].W[j][v]=ww-eta
                        loss2=self.compute_loss(data_x,data_y)
                        self.layers[i].W[j][v]=ww
                        dw_=np.hstack((dw_,(loss1-loss2)/(2*eta)))
                for j in range(self.layers[i].W.shape[0]):
                    bb=self.layers[i].b[0][j]
                    self.layers[i].b[0][j]=bb+eta
                    #set_trace()
                    loss1=self.compute_loss(data_x,data_y)
                    self.layers[i].b[0][j]=bb-eta
                    loss2=self.compute_loss(data_x,data_y)
                    self.layers[i].b[0][j]=bb
                    dw_=np.hstack((dw_,(loss1-loss2)/(2*eta)))
            if self.layers[i].name == 'conv': 
                for j in range(self.layers[i].W.shape[0]):
                    for k in range(self.layers[i].W.shape[1]):
                        for l in range(self.layers[i].W.shape[2]):
                            for v in range(self.layers[i].W.shape[3]):
                                ww=self.layers[i].W[j][k][l][v]
                                self.layers[i].W[j][k][l][v]=ww+eta
                                loss1=self.compute_loss(data_x,data_y)
                                self.layers[i].W[j][k][l][v]=ww-eta
                                loss2=self.compute_loss(data_x,data_y)
                                self.layers[i].W[j][k][l][v]=ww
                                dw_=np.hstack((dw_,(loss1-loss2)/(2*eta)))
                for j in range(self.layers[i].b.shape[0]):
                    bb=self.layers[i].b[j]
                    self.layers[i].b[j]=bb+eta
                    #set_trace()
                    loss1=self.compute_loss(data_x,data_y)
                    self.layers[i].b[j]=bb-eta
                    loss2=self.compute_loss(data_x,data_y)
                    self.layers[i].b[j]=bb
                    dw_=np.hstack((dw_,(loss1-loss2)/(2*eta)))
        #set_trace()
        return np.vstack((dw.T,dw_.T)).T
    def fit(self,data_x,data_y,min_batch_size,epoch_num,optf):
        datasize=data_x.shape[0]
        layersnum=len(self.layers)

        #construct dw_g,db_g
        dW_g =[]
        db_g =[]
        for i in np.arange(layersnum):
            if self.layers[i].name != 'pool':
                dW_g.append(np.zeros([len(self.layers[i].dW.flatten())]))
                db_g.append(np.zeros([len(self.layers[i].db.flatten())]))
            else:
                dW_g.append(np.zeros([100]))
                db_g.append(np.zeros([100]))
        for i in range(epoch_num):
            order=np.arange(datasize)
            np.random.shuffle(order)
            data_x=data_x[order]
            data_y=data_y[order]
            #set_trace()
            for j in range(int(data_x.shape[0]/min_batch_size)):
                begin=j*min_batch_size
                end=(j+1)*min_batch_size
                self.backwardpropagation(data_x[begin:end],data_y[begin:end])
                for v in np.argsort(-np.arange(layersnum)):
                    if self.layers[v].name != 'pool':
                        #self.layers[v].W=optf.updateData(self.layers[v].W.flatten(),self.layers[v].dW.flatten()).reshape(netshape[v+1],netshape[v]+1)
                        W,dW=optf.updateData(self.layers[v].W.flatten(),self.layers[v].dW.flatten(),dW_g[v])
                        dW_g[v]=dW
                        b,db=optf.updateData(self.layers[v].b.flatten(),self.layers[v].db.flatten(),db_g[v])
                        db_g[v]=db
                        self.layers[v].W=W.reshape(self.layers[v].W.shape)
                        self.layers[v].dW=dW.reshape(self.layers[v].dW.shape)
                        self.layers[v].b=b.reshape(self.layers[v].b.shape)
                        self.layers[v].db=db.reshape(self.layers[v].db.shape)
                cost_loss_batch=self.compute_loss(data_x[begin:end],data_y[begin:end])
                print('batch %d :batch_loss=%f'%(j,cost_loss_batch))
            cost_loss=self.compute_loss(data_x,data_y)
            print('epoch %d :loss=%f'%(i,cost_loss))
        #return cost_loss
    def predict(self,data_x):
        yy=self.forwardpropagation(data_x)
        datasize=data_x.shape[0]
        y=np.zeros([datasize,1])
        for i in range(datasize):
            for j in range(yy[i].size):
                if np.max(yy[i])==yy[i][j]:
                    y[i]=j
        return y
    def compute_predict_rate(self,data_x,data_y):
        r=self.predict(data_x)
        cond=(r==data_y)
        rn=np.where(cond,1.0,0.0)
        rate=rn.sum()/rn.size
        #set_trace
        return rate
