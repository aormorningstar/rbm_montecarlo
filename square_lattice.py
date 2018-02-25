# square_lattice.py
# source code for square_lattice class
# Alan Morningstar
# February 2018

import numpy as np

# a square lattice class
class square_lattice:

        # initialize rbm
        def __init__(self,L,pbc = True):

            # length in each dimension
            self.L = L
            # boundary conditions
            self.pbc = pbc
            # number of sites
            self.n = L*L
            # vectors from site to neighboring sites
            self.nbrVecs = [np.array([1,0]),np.array([0,1]),np.array([-1,0]),np.array([0,-1])]

            # list of neighboring sites
            self.nbrs = []
            self.numNbrs = []
            for s in range(0,self.n):
                # xy coords
                xyVec = self.xyIndex(s)
                xyNbrs = [xyVec + vec for vec in self.nbrVecs]
                sNbrs = []

                # add site indexed neighbors to list
                if self.pbc:
                    for xy  in xyNbrs:
                        sNbrs.append(self.sIndex(xy))
                else:
                    for xy in xyNbrs:
                        if (xy[0] >= 0) and (xy[0] < self.L) and (xy[1] >= 0) and (xy[1] < L):
                            sNbrs.append(self.sIndex(xy))

                self.nbrs.append(sNbrs)
                self.numNbrs.append(len(sNbrs))

        # convert from site index s to xy position on the lattice
        # xy indexing starts with (0,0) in the bottom left corner of the lattice
        def xyIndex(self,s):
            x = s+1
            y = 1

            while x > self.L:
                x -= self.L
                y += 1

            # return x,y index
            return np.array([x-1,y-1])

        # convert from xy position on lattice to site index
        # s indexing starts with 0 in the bottom left corner of the lattice
        def sIndex(self,xy):
            return self.L*(xy[1]%self.L) + (xy[0]%self.L)
