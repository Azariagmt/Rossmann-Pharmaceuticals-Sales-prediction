import numpy as np
def rmspe(y, yhat):
    '''
    Loss function for model evaluation
    '''
    rmspe = np.sqrt(np.mean((y - yhat)**2))
    return rmspe
