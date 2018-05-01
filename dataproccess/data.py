import numpy as np
import struct

def loadmnistImages(filedir):
    f=open(filedir,'rb')
    buf=f.read()
    index = 0
    magic,numImages,numRows,numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    images=np.zeros([numImages,numRows+4,numColumns+4])
    image_size = numRows*numColumns
    fmt_image = '>' + str(image_size) + 'B'
    for i in range(numImages):
        image=struct.unpack_from(fmt_image,buf,index)
        index+=struct.calcsize(fmt_image)
        images[i,2:30,2:30]=np.array(image).reshape(numRows,numColumns)
    return images
def loadmnistLabels(filedir):
    f=open(filedir,'rb')
    buf=f.read()
    index = 0
    magic,numlabels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labels=np.zeros([numlabels,1])
    for i in range(numlabels):
        label=struct.unpack_from('>B',buf,index)
        index+=struct.calcsize('>B')
        labels[i]=label
    return labels
