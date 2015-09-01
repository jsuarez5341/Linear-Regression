#Gradient Descent
import numpy as np
import scipy as sp

class GradientDescent(object):

    def doMinimize(self, gradTol, maxIterations):
        sumSqrErr=999999999
        grad=999999999
        iters=1
        params=np.zeros((1,self.numfeatures))
        alpha=1
        while (np.sum(abs(grad))>gradTol) & (iters<maxIterations):
            err=self.dataset[:, -1] - np.sum(self.dataset[:, 0:self.numfeatures]*params, 1)
            sumSqrErr=np.sum(.5*(err**2))
            grad=np.dot( (self.dataset[:,0:self.numfeatures]).transpose(), err ) #vector of parameter updates
            #update params
            if np.sum( .5*(self.dataset[:, -1] - np.sum(self.dataset[:, 0:self.numfeatures]* (params + alpha*grad), 1))**2 ) <sumSqrErr:
                params = params + alpha*grad
            else:
                #Cut alpha in half and toss the bad update
                alpha = alpha/2
        return params

    def __init__(self, dataset, gradTol, maxIterations):
        self.dataset = dataset #Must be a labeled dataset
        self.numfeatures= (np.shape(dataset)).__getitem__(1)-1   #num cols -1
        self.samples= (np.shape(dataset)).__getitem__(1)         #num rows
        self.optimizedParams = self.doMinimize(gradTol, maxIterations)







