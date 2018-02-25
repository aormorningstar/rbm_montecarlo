# exact_Ising_rbm.py
# source code for exact_Ising_rbm class
# Alan Morningstar
# February 2018

import numpy as np
import pandas as pd
from square_lattice import square_lattice

class exact_Ising_rbm(square_lattice):

    def __init__(self,T,lattice,hamiltonian):

        # lattice
        self.l = lattice

        # Hamiltonian
        self.H = hamiltonian

        # temperature
        self.T = T

        # network
        self.nV = self.l.n
        self.nH = 0

        # initialize weights (first column and row are biases)
        self.w = np.zeros((self.nV+1,self.nH+1))

        # set transposed weights
        self.wT = self.w.transpose()

        # set up exact parameter values
        self.exact_params()

    # fill in the exact parameter values
    def exact_params(self):

        # keep track of which hidden unit we're on
        j = 0
        # run over all sites
        for i1 in range(0,self.nV):

            # set visible biases
            self.w[i1+1,0] = -2.0*self.H.hs[i1]/self.T

            # run over bonds to neighbors
            for i2Index,i2 in enumerate(self.l.nbrs[i1]):
                if i1 > i2:
                    # make a new hidden unit
                    self.w = np.append(self.w,np.zeros((self.nV+1,1)),axis=1)

                    # common factors for exact parameters
                    w0 = 0.5*np.arccosh(np.exp(2.0*abs(self.H.Js[i1][i2Index])/self.T))
                    signJ = np.sign(self.H.Js[i1][i2Index])

                    # set exact weights
                    self.w[i1+1,j+1] = w0
                    self.w[i2+1,j+1] = signJ*w0

                    # increment hidden unit counter
                    j += 1

        self.wT = self.w.transpose()

        print("Total number of hidden neurons is ",j,"\n")

    # load already trained model from file
    def loadModel(self,modelFileName,verb = True):
        if verb:
            print('----------------------------------')
            print('loading model from ',modelFileName)

        # load model parameters into data frame
        modelDF = pd.read_csv(modelFileName,sep=',',header=None,index=None)

        # set weights
        self.w = modelDF.values
        self.wT = modelDF.values.transpose()

    # compute hidden probabilities from hidden state
    def hiddenProbs(self,vState):
        # update hProbs
        hProbs = self.activate(np.dot(vState,self.w))
        # set hidden units corresponding to biases to 1
        hProbs[:,0] = 1.0
        return hProbs

    # sample hidden state from hidden probabilities
    def hiddenState(self,hProbs):
        return 2*(hProbs > np.random.uniform(0.0,1.0,hProbs.shape)).astype(int)-1

    # compute visible probabilities from hidden state
    def visibleProbs(self,hState):
        # update vProbs
        vProbs = self.activate(np.dot(hState,self.wT))
        # set visible units corresponding to biases to 1
        vProbs[:,0] = 1.0
        return vProbs

    # sample visible state from visible probabilities
    def visibleState(self,vProbs):
        return 2*(vProbs > np.random.uniform(0.0,1.0,vProbs.shape)).astype(int)-1

    # gibbs cycle
    def gibbshvh(self,hState):
        # gibbs cycle, reconstruct hState, update visibles and hiddens
        vP = self.visibleProbs(hState)
        vS = self.visibleState(vP)
        hP = self.hiddenProbs(vS)
        hS = self.hiddenState(hP)
        return vP,vS,hP,hS

    # gibbs cycle
    def gibbsvhv(self,vState):
        # gibbs cycle, reconstruct vState, update visibles and hiddens
        hP = self.hiddenProbs(vState)
        hS = self.hiddenState(hP)
        vP = self.visibleProbs(hS)
        vS = self.visibleState(vP)
        return hP,hS,vP,vS

    # infer h state given v state
    def vh(self,vState):
        return self.hiddenState(self.hiddenProbs(vState))

    # infer v state given h state
    def hv(self,hState):
        return self.visibleState(self.visibleProbs(hState))

    # save the network parameters
    def saveModel(self,modelFileName):
        # save weights and biases to csv file
        wDF = pd.DataFrame(self.w)
        wDF.to_csv(modelFileName,sep=',',header=False,index=False)

    # sample the rbm
    def samples(self,numSamples,numCycles,verb = True):
        if verb:
            print('----------------------------------')
            print('Sampling rbm ...')

        # matrices for storing probabilities and states, initialize visibles randomly
        hShape = (numSamples,self.nH+1)
        vShape = (numSamples,self.nV+1)
        hProbs = np.zeros(hShape)
        hState = np.zeros(hShape)
        vProbs = np.random.uniform(0.0,1.0,vShape)
        vProbs[:,0] = 1.0
        vState = self.visibleState(vProbs)

        # cycle markov chain
        for i in range(numCycles):
            hProbs,hState,vProbs,vState = self.gibbsvhv(vState)

        return vState[:,1:]

    # utils
    def activate(self,x):
        return 1.0/(1.0 + np.exp(-2*x))
