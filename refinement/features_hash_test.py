import numpy as np
import torch
import os

path = '/home/user/wangqx/stanford/Learning_based_method/data0'
feature0 = np.load(os.path.join(path, 'features_0_0.npy'))
feature1 = np.load(os.path.join(path, 'features_0_1.npy'))
feature2 = np.load(os.path.join(path, 'features_0_2.npy'))
feature3 = np.load(os.path.join(path, 'features_0_3.npy'))

feat_tuple0 = [tuple(x) for x in feature0.tolist()]
feat_tuple1 = [tuple(x) for x in feature1.tolist()]
feat_tuple2 = [tuple(x) for x in feature2.tolist()]
feat_tuple3 = [tuple(x) for x in feature3.tolist()]

feat_hash0 = [hash(x) for x in feat_tuple0]
feat_hash1 = [hash(x) for x in feat_tuple1]
feat_hash2 = [hash(x) for x in feat_tuple2]
feat_hash3 = [hash(x) for x in feat_tuple3]
feat_hash = feat_hash0 + feat_hash1 + feat_hash2 + feat_hash3

from collections import Counter
counts = Counter(feat_hash)
num_non_unique = sum(count > 1 for count in counts.values())

print('num_non_unique: ', num_non_unique)

key = 20
features = np.load(f'../data/bear_features{key}.npy')
feat_tuple = [tuple(x) for x in features.tolist()]
feat_hash = [hash(x) for x in feat_tuple]
from collections import Counter
counts = Counter(feat_hash)
num_non_unique = sum(count > 1 for count in counts.values())
print('num_non_unique: ', num_non_unique)
