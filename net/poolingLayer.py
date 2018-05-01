import numpy as np
from activation.activation import acti_func


def downsample(inmap,kernelsize,type):
    inmapH,inmapW = inmap.shape
    outmapH=int(inmapH/kernelsize+(1 if inmapH%kernelsize>0 else 0))
    outmapW=int(inmapW/kernelsize+(1 if inmapW%kernelsize>0 else 0))
    outmap=np.zeros([outmapH,outmapW])
    flagmap=np.zeros_like(inmap)
    for i in range(outmapH):
        for j in range(outmapW):
            if(type == 'max'):
                outmap[i,j]=np.max(inmap[i*kernelsize:(i+1)*kernelsize,j*kernelsize:(j+1)*kernelsize])
                idx=np.argmax(inmap[i*kernelsize:(i+1)*kernelsize,j*kernelsize:(j+1)*kernelsize])
                idx_h=int(idx/kernelsize)
                idx_w=idx%kernelsize
                flagmap[i*kernelsize+idx_h,j*kernelsize+idx_w]=1
            if(type == 'mean'):
                outmap[i,j]=np.mean(inmap[i*kernelsize:(i+1)*kernelsize,j*kernelsize:(j+1)*kernelsize])
                flagmap[i*kernelsize:(i+1)*kernelsize,j*kernelsize:(j+1)*kernelsize]=1/(kernelsize*kernelsize)
    return outmap,flagmap
def upsample(inmap,kernelsize,flagmap,type):
    inmapH,inmapW = inmap.shape
    outmapH=inmapH*kernelsize
    outmapW=inmapW*kernelsize
    outmap=np.zeros([outmapH,outmapW])
    for i in range(inmapH):
        for j in range(inmapW):
            outmap[i*kernelsize:(i+1)*kernelsize,j*kernelsize:(j+1)*kernelsize]=inmap[i,j]*np.ones([kernelsize,kernelsize])
    outmap=outmap*flagmap
    return outmap
class PoolingLayer(object):
    def __init__(self,inputH,inputW,kernelsize,inchannels,type,acti,name='pool'):
        self.inputH = inputH
        self.inputW = inputW
        self.outputH = int(inputH/kernelsize)
        self.outputW = int(inputW/kernelsize)
        self.kernelsize = kernelsize
        self.inchannels = inchannels
        self.outchannels = inchannels
        self.type = type
        self.acti = acti_func('identity')
        self.name = name
        self.inputdata = np.zeros([1,self.inchannels,self.inputH,self.inputW])
        self.inputflags = np.zeros([1,self.inchannels,self.inputH,self.inputW])
        #print([1,self.inchannels,self.outputH,self.outputW])
        self.outputdata = np.zeros([1,self.inchannels,self.outputH,self.outputW])
        self.inputdeltas = np.zeros([1,self.inchannels,self.inputH,self.inputW])
        self.outputdeltas = np.zeros([1,self.inchannels,self.outputH,self.outputW])
    def forwardpropagation(self,inputmaps):
        self.inputdata=inputmaps
        self.inputflags=np.zeros_like(inputmaps)
        batchsize,inchannels,inH,inW=inputmaps.shape
        outH=int(inH/self.kernelsize+(1 if inH%self.kernelsize>0 else 0))
        outW=int(inW/self.kernelsize+(1 if inW%self.kernelsize>0 else 0))
        self.outputmaps=np.zeros([batchsize,self.outchannels,outH,outW])
        for i in range(batchsize):
            for j in range(self.outchannels):
                self.outputmaps[i,j],self.inputflags[i,j]=downsample(inputmaps[i,j],self.kernelsize,self.type)
                self.outputmaps[i,j]=self.acti.f(self.outputmaps[i,j])
        self.outputdata=self.outputmaps
        return self.outputmaps
    def backwardpropagation(self,inputmaps,outputdeltas,prev_acti):
        batchsize,inchannels,inH,inW = inputmaps.shape
        self.inputdeltas=np.zeros([batchsize,self.inchannels,self.inputH,self.inputW])
        self.outputdeltas=np.zeros([batchsize,self.inchannels,self.outputH,self.outputW])
        self.outputdeltas=outputdeltas	
        for i in range(batchsize):
            for j in range(self.inchannels):
                self.inputdeltas[i,j]=upsample(outputdeltas[i,j],self.kernelsize,self.inputflags[i,j],self.type)
                self.inputdeltas[i,j]=self.inputdeltas[i,j]*prev_acti.f_(inputmaps[i,j])
        return self.inputdeltas
