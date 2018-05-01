import sys
sys.path.append("..")

import numpy as np
from activation.activation import acti_func


def Conv2(inmap,kernel,type):
    inmapH,inmapW=inmap.shape
    kernelH,kernelW=kernel.shape
    if type == 'valid':
        outmapH = inmapH-kernelH+1
        outmapW = inmapW-kernelW+1		
    if type == 'full':
        outmapH = inmapH+kernelH-1
        outmapW = inmapW+kernelH-1
        #inmap=np.column_stack(inmap,np.zeros([inmapH,kernelW-1]))
        #inmap=np.row_stack(inmap,np.zeros([kernelH-1,inmapW+kernelW-1]))
    inmap_=np.zeros([inmapH+2*(kernelH-1),inmapW+2*(kernelW-1)])
    inmap_[kernelH-1:inmapH+(kernelH-1),kernelW-1:inmapW+(kernelW-1)]=inmap
    inmap=inmap_		
    outmap=np.zeros([outmapH,outmapW])
    for i in range(outmapH):
	    for j in range(outmapW):
	        outmap[i,j]+=np.sum(inmap[i:i+kernelH,j:j+kernelW]*kernel)
    return outmap
def rot180(inmap):
    inmapH,inmapW = inmap.shape
    for i in range(int(inmapH/2)):
        for j in range(inmapW):
            inmap[i,j]=inmap[inmapH-1-i,j]
    for i in range(inmapH):
        for j in range(int(inmapW/2)):
            inmap[i,j]=inmap[i,inmapW-1-j]
    return inmap
def Conv2_(inmap,kernel,type):#np函数实现
    return np.convolve(kernel,inmap,tyep)
def rot180_(inmap):#np函数实现
    return np.rot90(inmap,2)
class ConvolutionLayer(object):
    def __init__(self,inputH,inputW,kernelsize,inchannels,outchannels,acti,name='conv'):
        self.inputH = inputH
        self.inputW = inputW
        self.kernelsize = kernelsize
        self.outputH=inputH-self.kernelsize+1
        self.outputW=inputW-self.kernelsize+1
        self.inchannels=inchannels
        self.outchannels = outchannels
        self.W=np.zeros([self.inchannels,self.outchannels,self.kernelsize,self.kernelsize])
        self.dW=np.zeros_like(self.W)
        self.b=np.zeros(self.outchannels)
        self.db=np.zeros_like(self.b)
        self.inputdata=np.zeros([1,self.inchannels,self.inputH,self.inputW])
        self.outdata=np.zeros([1,self.inchannels,self.outputH,self.outputW])

        self.inputdeltas=np.zeros([1,self.inchannels,self.inputH,self.inputW])
        self.outputdeltas=np.zeros([1,self.outchannels,self.outputH,self.outputW])
        self.acti = acti
        self.name = name

        base=np.sqrt(6.0/(self.inchannels+self.outchannels))
        self.W=(np.random.random(self.W.size)*2*base-base).reshape(self.inchannels,self.outchannels,self.kernelsize,self.kernelsize)
        self.b=(np.random.random(self.b.size)*2*base-base).reshape(self.outchannels)

    def forwardpropagation(self,inputmaps):
        self.inputdata=inputmaps
        batchsize,inchannels,inH,inW=inputmaps.shape
        outH=inH-self.kernelsize+1
        outW=inW-self.kernelsize+1
        outputmaps=np.zeros([batchsize,self.outchannels,outH,outW])
        for i in range(batchsize):
            for j in range(self.outchannels):
                for k in range(self.inchannels):				
                    outputmaps[i,j]+=Conv2(inputmaps[i,k],self.W[k,j],'valid')
                    outputmaps[i,j]+=self.b[j]
                    outputmaps[i,j]=self.acti.f(outputmaps[i,j])
        self.outdata=outputmaps
        return self.outdata
    def backwardpropagation(self,inputmaps,outputdeltas,prev_acti):	
        batchsize,inchannels,inH,inW = inputmaps.shape
        self.inputdeltas=np.zeros([batchsize,self.inchannels,self.inputH,self.inputW])
        self.outputdeltas=np.zeros([batchsize,self.outchannels,self.outputH,self.outputW])
        self.outputdeltas=outputdeltas
        for i in range(self.inchannels):
            for j in range(self.outchannels):
                for k in range(batchsize):
                    self.dW[i,j]+=Conv2(inputmaps[k,i],self.outputdeltas[k,j],'valid')
                    self.db[j]+=np.sum(self.outputdeltas[k,j])
        self.dW=self.dW/batchsize
        self.db=self.db/batchsize
        for i in range(batchsize):
            for j in range(self.inchannels):
                for k in range(self.outchannels):
                    self.inputdeltas[i,j]+=rot180(Conv2(rot180(self.outputdeltas[i,k]),self.W[j,k],'full'))
                    #self.inputdeltas[i,j]+=Conv2(self.outputdeltas[i,k],rot180(self.W[j,k]),'full')
                    self.inputdeltas[i,j]=self.inputdeltas[i,j]*prev_acti.f_(inputmaps[i,j])
        return self.inputdeltas

