#Main

import numpy as np
import scipy as sp
from GradientDescentTest.Dataset import Dataset
from GradientDescentTest.GradientDescent import GradientDescent

if __name__ == '__main__':

    dataset=Dataset(3, 5, (1,2,3), (3,5,7))
    print(dataset.data)

    optimizedParams=GradientDescent(dataset.data, .00001, 500).optimizedParams
    print(optimizedParams)

    #Correct values to return are ([1 1.25 1.5])
    #These values are listed and editable in Dataset
