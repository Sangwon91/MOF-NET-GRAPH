import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

from mofnet import MofNet

mofnet = MofNet(32, 32)

nt = np.zeros(shape=[1, 1], dtype=np.int32)
nl = np.zeros(shape=[1, 1, 1], dtype=np.int32)
et = np.zeros(shape=[1, 1, 1], dtype=np.int32)
st = np.zeros(shape=[1, 1, 32], dtype=np.float32)

mofnet(nt, nl, et, st)
mofnet.load_weights("MOF-50000-sa-1.h5")

mofnet.summary()
