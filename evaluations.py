import numpy as np
from skimage.segmentation import find_boundaries


def achievable_segmentation_accuracy(superpixel, label):
    '''
    Function to calculate Achievable Segmentation Accuracy:
        ASA(S,G) = sum_j max_i |s_j \cap g_i| / sum_i |g_i|

    Args:
        input: superpixel image (H, W),
        output: ground-truth (H, W)
    '''
    TP = 0
    unique_id = np.unique(superpixel)
    for uid in unique_id:
        mask = superpixel == uid
        label_hist = np.histogram(label[mask])
        maximum_regionsize = label_hist[0].max()
        TP += maximum_regionsize
    return TP / (label.shape[0]*label.shape[1])


def undersegmentation_error(superpixel, label):
    '''
    To calculate undersegmentation error
        Undersegmentation error = 1/N sum(sum(min(Pin,Pout)))
    Args:
    superpixel: superpixel segmentation computed by some method 
    label:      semantic segmentation ground truth
    output:
        undersegmentation error astype float
    '''
    outliners = 0
    unique_id = np.unique(superpixel)
    for uid in unique_id:
        mask = superpixel == uid
        label_hist = np.histogram(label[mask])
        outliner = label[mask].shape[0] - label_hist[0].max()
        outliners += outliner
    return outliners / (label.shape[0]*label.shape[1])


def boundary_circumference(superpixel):
    '''
    Cal the circumference of an 2D superpixel,numpy array
    input: a superpixel lied in picture
    output: 
        the circumference of superpixel astype int
        the area of superpixel astype int

    need:
        skimage.segmentation.find_boundaries
        numpy
    '''
    boundary = find_boundaries(superpixel, mode='inner')
    round = np.sum(boundary == 1)
    mask = np.asarray(np.where(boundary == 1)).transpose()
    hidx = np.where(mask[:, 0] == 0)
    max = 0
    min = 0
    # cal the length of horizontial image boundary for superpixels that reaches the boundary
    for _, single_idx in enumerate(hidx[0]):
        if mask[single_idx, 1] > max:
            max = mask[single_idx, 1]
        if mask[single_idx, 1] < min and mask[single_idx, 1] < max-1:
            min = mask[single_idx, 1]
    hori = max-min
    vidx = np.where(mask[:, 1] == 0)
    # find the
    max = min = 0
    for _, single_idx in enumerate(vidx[0]):
        if mask[single_idx, 0] > max:
            max = mask[single_idx, 0]
        if mask[single_idx, 0] < min and mask[single_idx, 0] < max-1:
            min = mask[single_idx, 0]
    verti = max-min
    #  Ls               As
    return float(round + hori + verti), np.sum(superpixel == 1, dtype=float)


def compactness(segmentation):
    '''
    cal the compactness of a superpixel segmentation
    input: 
        segmentation:the result of superpixel segmentation
    output:
        compactness: as the name means
    '''
    # for each superpixel
    cptness = 0.0
    for _,     i in enumerate(np.unique(segmentation)):
        mask = np.where(segmentation == i)
        subimage = np.zeros(segmentation.shape)
        subimage[mask] = 1
        Ls, As = boundary_circumference(subimage)
        # Qs*|S|  Qs = As/Ac = 4pi*As/Ls^2,As = S in this case
        cptness += (As / Ls) * (As / Ls)
        # then the formula is 4pi*(As/Ls)^2
    return 4*np.pi*cptness/(segmentation.shape[0]*segmentation.shape[1])


def boundary_recall(segmentation, groundTruth, connectivity8=False):
    '''
    Compute Boundary Reall for a given Superpixel and groundtruth
    input:
        segmentation:   segmented superpixels
        groundTruth:    semantic groundtruth,Maybe upgrate to boundary using some boundary detection Algorithm
        connectivity8:  using 8 connectivity instead of 4
    output:
        boundary_recall: boundary recall astype float
    '''
    def is4Connective(gt, x, y):
        '''
        find if there is a boundary near a given coordinate
        input:
            gt: boundary of spix
            x,y:an coordinate
        output:
            bool to show if there is a boundary in 4 connective field
        '''
        assert x >= 0 and x < gt.shape[0] and y >= 0 and y < gt.shape[1], (
            x, y, gt.shape[0], gt.shape[1])

        if(gt[x][y] == 1):
            return True
        if(x > 0 and gt[x-1][y] == 1):
            return True
        if(y > 0 and gt[x][y-1] == 1):
            return True
        if(x < gt.shape[0]-1 and gt[x+1][y] == 1):
            return True
        if(y < gt.shape[1]-1 and gt[x][y+1] == 1):
            return True

        return False

    def is8Connective(gt, x, y):
        '''
        find if there is a boundary in 4 connective field a given coordinate
        input:
            gt: boundary of spix
            x,y:an coordinate
        output:
            bool to show if there is a boundary in 4 connective field
        '''
        if(is4Connective(gt, x, y)):
            return True
        if(x > 0 and y > 0 and gt[x-1][y-1] == 1):
            return True
        if(x < gt.shape[0]-1 and y > 0 and gt[x+1][y-1] == 1):
            return True
        if(x > 0 and y < gt.shape[1]-1 and gt[x][y+1] == 1):
            return True
        if(x < gt.shape[0]-1 and y < gt.shape[1]-1 and gt[x+1][y+1] == 1):
            return True

        return False

    if(connectivity8):
        conncetivity = is8Connective
    else:
        conncetivity = is4Connective

    boundary = find_boundaries(groundTruth, mode='inner')
    spix_boundary = find_boundaries(segmentation, mode='inner')
    mask = np.asarray(np.where(boundary == 1)).transpose()
    recalled = 0.0
    for i in range(mask.shape[0]):
        if(conncetivity(spix_boundary, mask[i][0], mask[i][1])):
            recalled += 1
    return recalled/np.sum(boundary)
