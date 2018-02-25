# Ising_measure.py
# source code for Ising_measure class
# Alan Morningstar
# February 2018

import numpy as np
import pandas as pd

class Ising_measure():

    def __init__(self,Ising_rbm,numSamples,numCycles):

        self.Ising_rbm = Ising_rbm
        self.numSamples = numSamples
        self.numCycles = numCycles
        self.data = pd.DataFrame(self.Ising_rbm.samples(self.numSamples,self.numCycles))

        self.m = None
        self.chi = None
        self.e = None
        self.c = None

    # compute magnetization and susceptibility (both per spin)
    def magnetization(self):

        nV = self.Ising_rbm.nV
        T = self.Ising_rbm.T
        L = self.Ising_rbm.l.L

        # compute magnetization (m) and susceptibility (chi)
        mData = abs(self.data.sum(1))/nV
        mSquaredData = mData**2

        self.m = mData.sum()/self.numSamples
        mSquared = mSquaredData.sum()/self.numSamples
        self.chi = nV*(mSquared - self.m**2.0)/T

        print('Temperature    = ',T)
        print('Magnetization  = ',self.m)
        print('Susceptibility = ',self.chi)

    # compute energy and heat capacity (both per spin)
    def energy(self):

        nV = self.Ising_rbm.nV
        T = self.Ising_rbm.T
        L = self.Ising_rbm.l.L
        J = self.Ising_rbm.H.J
        h = self.Ising_rbm.H.h

        # compute energy per spin (e) and heat capacity (c)
        eList = np.zeros(self.numSamples)
        for i in range(self.numSamples):
            s = self.data.values[i,:].reshape((L,L))
            eiSum = 0
            for j in range(L):
                for k in range(L):
                    eiSum -= s[j,k]*(s[(j+1)%L,k]+s[j-1,k]+s[j,(k+1)%L]+s[j,k-1])

            eList[i] = 0.5*eiSum/nV

        self.e = np.sum(eList)/self.numSamples
        eSquared = np.sum(eList*eList)/self.numSamples
        self.c = nV*(eSquared - self.e*self.e)/(T*T)

        print('Energy Per Spin = ',self.e)
        print('Specific Heat = ',self.c)
