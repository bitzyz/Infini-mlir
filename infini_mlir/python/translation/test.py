import numpy  as np
data = np.load('resnet_origin_weight.npz')
print(data.files)
print(data['resnetv22_dense0_weight_fix'])