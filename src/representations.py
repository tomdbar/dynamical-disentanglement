import torch
import math
from collections import defaultdict

class Representation():

    def __init__(self, dim=4, device="cpu"):
        self.dim = dim
        self.device = device
        self.params = dim * (dim - 1) // 2
        self.thetas = torch.autograd.Variable(math.pi*(2*torch.rand(self.params, device=self.device)-1) / dim,
                                              requires_grad=True)

        self.clear_matrix()

    def set_thetas(self, thetas):
        self.thetas = thetas.to(self.device)
        self.thetas.requires_grad = True
        self.clear_matrix()

    def clear_matrix(self):
        '''Clear the cached unitary matrix.

        The action matrix is cached to avoid re-calculating them at every step.  However,
        if the underlying thetas are changed (e.g. after a step of SGD), this cache must
        be cleared so that the correct matrix is re-calculated and cached in its place.
        '''
        self.__matrix = defaultdict(lambda: None)

    def get_matrix(self, magnitude=1):
        if self.__matrix[magnitude] is None:
            k = 0
            mats = []
            for i in range(self.dim - 1):
                for j in range(self.dim - 1 - i):
                    theta_ij = self.thetas[k] * magnitude
                    k += 1
                    c, s = torch.cos(theta_ij), torch.sin(theta_ij)

                    rotation_i = torch.eye(self.dim, self.dim)
                    rotation_i[i, i] = c
                    rotation_i[i, i + j + 1] = s
                    rotation_i[j + i + 1, i] = -s
                    rotation_i[j + i + 1, j + i + 1] = c

                    mats.append(rotation_i)

            def chain_mult(l):
                if len(l) >= 3:
                    return l[0] @ l[1] @ chain_mult(l[2:])
                elif len(l) == 2:
                    return l[0] @ l[1]
                else:
                    return l[0]

            self.__matrix[magnitude] = chain_mult(mats).to(self.device)

        return self.__matrix[magnitude]