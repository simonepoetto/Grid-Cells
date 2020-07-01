import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from gtda.homology import CubicalPersistence

from persim import plot_diagrams

from cv2 import GaussianBlur

from visualize import plot_ratemaps
from scores import GridScorer


class Activations():
    
    def __init__(self, ratemaps):
        self.ratemaps_ = ratemaps
        self.res_ = self.ratemaps_.shape[1]
        self.len_ = self.ratemaps_.shape[0]
        
        
    def smooth_rm(self, ksize=(3,3), sigmaX=1, sigmaY=0):
        for i in range(self.len_):
            self.ratemaps_[i] = GaussianBlur(self.ratemaps_[i], ksize, sigmaX, sigmaY)
        

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
        for rm in tqdm(self.ratemaps_):
            sac = self._calculate_sac(rm,mask)
            sacslist.append(sac)
            sacs = np.stack(sacslist)
        self.sacs_ = sacs

        return sacs
    
    def calculate_dgms(self, digits=64):
        sacs = self.sacs_.round(digits)
        cubpers = CubicalPersistence(infinity_values=1,n_jobs=-1)
        self.dgms_ = cubpers.fit_transform(-sacs)
        return self.dgms_
    
    
    def plot_rm_sac_dgm(self,idx):
        plt.figure(figsize=(15,4))
        plt.subplot(131)
        plt.imshow(self.ratemaps_[idx],cmap='jet')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(self.sacs_[idx],cmap='jet')
        plt.axis('off')
        #convert dgm for sktda
        dgmskt = []
        for j in list(set(self.dgms_[idx,:,2])):
            h = self.dgms_[idx][np.where(self.dgms_[idx][:,2]==j),:2].reshape(-1,2)
            dgmskt.append(h)
        plt.subplot(133)
        plot_diagrams(dgmskt)
        
    
    def vizall(self, images, n_plot=None,cmap='jet', smooth=False, n_col=16):
        if not n_plot:
            n_plot = images.shape[0]
        plt.figure(figsize=(16,4*n_plot//8**2))
        rm_fig = plot_ratemaps(images, n_plot, cmap, smooth, n_col)
        plt.imshow(rm_fig)
        plt.axis('off');
        
    
    def scores(self,ratemaps='all', box_width=2.2, box_height=2.2):
        ''' Return
        score60 = vector of scores of 60° gridness dimension len ratemaps
        idxs = indexes of ratemaps in decreasing order respect to score
        '''
        if not isinstance(ratemaps,np.ndarray):
            ratemaps=self.ratemaps_
        res = self.res_
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
        masks_parameters = zip(starts, ends.tolist())
        scorer = GridScorer(res, coord_range, masks_parameters)
        score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(*[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(ratemaps)])
                                                                              
        idxs = np.flip(np.argsort(score_60))
        
        return score_60, idxs
    
    def viz_clusters(self,cut_tree, imgs, n_col=16):
        cut_tree = cut_tree[:,0]
        labels = np.unique(cut_tree)
        clusters = []
        for l in labels:
            clusters.append(imgs[np.where(cut_tree==l)])
            print('')
            print('Cluster'+str(l))
            self.vizall(clusters[l],n_col=n_col)
            plt.show()