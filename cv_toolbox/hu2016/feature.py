from ..features.shog import shog
from ..features.lbp_gradient import lbp_gradient
from ..features.glcm import sglcm
import numpy as np
from sklearn import preprocessing

def extract_feature_vector(img):
    _glbp = np.append(lbp_gradient(img, 8, 1), lbp_gradient(img, 16, 2)).astype('float')
    _sglcm = sglcm(img)
    _shog = shog(img)
    glbp_l1 = preprocessing.normalize(_glbp.reshape(1,-1), norm='l1').flatten()
    sglcm_l1 = preprocessing.normalize(_sglcm.reshape(1,-1), norm='l1').flatten()
    shog_l1 = preprocessing.normalize(_shog.reshape(1,-1), norm='l1').flatten()

    return np.concatenate((glbp_l1, sglcm_l1, shog_l1))
