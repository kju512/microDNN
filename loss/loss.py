import numpy as np

def MeanSquareErorrLoss(pred_y,data_y):
    r=-np.sum(np.mean(np.square(pred_y-data_y)))
    return r

def crossEntropyLoss(pred_y,data_y):
    r=-np.sum(np.mean(np.log(pred_y)*data_y+(1.0-data_y)*np.log(1.0-pred_y),0))
    return r

