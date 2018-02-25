# nn_Ising_Hamiltonian.py
# source code for nn_Ising_Hamiltonian class
# Alan Morningstar
# February 2018

import numpy as np
import copy as cp

# a nearest neighbor Ising Hamiltonian
class nn_Ising_Hamiltonian():

        # initialize Hamiltonian
        def __init__(self,lattice,J = 0.0, h = 0.0,randJ = False,randJmean = 0.0,randJstddev = 1.0,randh = False,randhmean = 0.0,randhstddev = 1.0):

            self.l = lattice

            # a list of length (number of sites), each entry contains a list of couplings from
            # a site to its neighbors organized according to lattice.nbrs
            self.Js = []
            # a list of length (number of sites), each entry is the magnetic field on that site
            self.hs = []

            # fill in Js, uniform or randomized
            if (not randJ):
                for site in range(self.l.n):
                    self.Js.append([J]*self.l.numNbrs[site])
            else:
                for site in range(self.l.n):
                    self.Js.append([randJstddev*np.randn()+randJmean for i in range(self.l.numNbrs[site])])

            # fill in hs, uniform or randomized
            if (not randh):
                for site in range(self.l.n):
                    self.hs.append(h)
            else:
                for site in range(self.l.n):
                    self.hs.append(randhstddev*np.randn()+randhmean)

            # def energy(self,config):
