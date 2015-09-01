#Dataset

'''
Dataset 2D Array Format
f=features, l=labels

fff...l
fff...l
.......
fff...l

'''

import numpy as np
import scipy as sp

class Dataset(object):

    def computeLabels(self, data):
        multipliers=np.linspace(1,2,5) #Produces as many of the following as needed: 1.0 1.25 1.5 1.75 2.0
        cols=(np.shape(data).__getitem__(1))
        data=data*multipliers[0:cols]
        data=np.sum(data,1)
        return data

    def __init__(self, numFeatures, numSamples, featureMins, featureMaxs):
        self.data=np.zeros((numSamples, numFeatures+1))
        self.data[:, 0:numFeatures] = (np.array(featureMaxs)-np.array(featureMins))*sp.rand(numSamples, numFeatures) + featureMins
        self.data[:, -1] = self.computeLabels(self.data[:, 0:numFeatures])




