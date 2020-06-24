import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from gtda.homology import CubicalPersistence

from persim import plot_diagrams

class Activations():
    
    def __init__(self, ratemaps):
        self.ratemaps = ratemaps
        self.res = self.ratemaps.shape[1]

    def _calculate_sac(self,rm, mask=1):
        
        '''
        Calculate sac for a single ratemap
        '''

        # define sac dim
        l = int(rm.shape[0] * mask)
        sac = np.zeros((l - 1, l - 1))
        sh = (l - 1) // 2
        
        rm = np.nan_to_num(rm)

        for i in range(-sh, sh + 1):
            for j in range(-sh, sh + 1):
                # shift ratemap
                shift = np.roll(np.roll(rm, i, axis=0), j, axis=1)

                # get row and column to remove
                if i == 0:
                    sx = slice(None)
                elif i > 0:
                    sx = slice(i, None)
                else:
                    sx = slice(None, i)
                if j == 0:
                    sy = slice(None)
                elif j > 0:
                    sy = slice(j, None)
                else:
                    sy = slice(None, j)

                rmr = rm[sx, sy]
                shiftr = shift[sx, sy]

                # calculate pearson correlation coefficient between flatten matrices
                rmf = np.ndarray.flatten(rmr)
                shiftf = np.ndarray.flatten(shiftr)

                pr = pearsonr(rmf, shiftf)

                sac[(l - 1) // 2 + i, (l - 1) // 2 + j] = pr[0]
        sac = np.nan_to_num(sac)
        return sac

    def calculate_sacs(self, mask=1):
        '''
        Return array of shape (n_ratemaps, sac_dim, sac_dim)
        containg the sac of every activation's ratemap
        '''
        sacslist = []
        for rm in tqdm(self.ratemaps):
            sac = self._calculate_sac(rm,mask)
            sacslist.append(sac)
            sacs = np.stack(sacslist)
        self.sacs = sacs

        return sacs
    
    def calculate_dgms(self):
        cubpers = CubicalPersistence(n_jobs=-1)
        self.dgms = cubpers.fit_transform(-self.sacs)
        return self.dgms
    
    
    def plot_rm_sac_dgm(self,idx):
        plt.figure(figsize=(15,4))
        plt.subplot(131)
        plt.imshow(self.ratemaps[idx],cmap='jet')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(self.sacs[idx],cmap='jet')
        plt.axis('off')
        #convert dgm for sktda
        dgmskt = []
        for j in list(set(self.dgms[idx,:,2])):
            h = self.dgms[idx][np.where(self.dgms[idx][:,2]==j),:2].reshape(-1,2)
            dgmskt.append(h)
        plt.subplot(133)
        plot_diagrams(dgmskt)

        