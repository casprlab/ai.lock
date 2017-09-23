import numpy as np
import h5py
import os


def store_dataset(dataset, out_file):
    h5_file = h5py.File(out_file, 'w')
    h5_file.create_dataset('dataset_1', data=dataset)
    h5_file.close()

layer_name = "bottleneck_FC"
# ----------------------------------------------------------------------------------------------------------------------
"""full size images"""
basic_path = "Datasets/" + layer_name + "/full_size"

# # Nexus dataset
# file = "Nexus.h5"
# hd = h5py.File(os.path.join(basic_path, file), 'r')
# dataset = hd['dataset_1']
# train = np.vstack((dataset[:800, :], dataset[1020:, :]))
# store_dataset(train, os.path.join(basic_path, "Nexus_train_1180.h5"))
# test = dataset[801:1020, :]
# store_dataset(train, os.path.join(basic_path, "Nexus_test_220.h5"))
# hd.close()

# ALOI: We have already shuffled the images when aggregating the embedding files. Now we just split
file = "ALOI.h5"
hd = h5py.File(os.path.join(basic_path, file), 'r')
dataset = hd['dataset_1']
train = dataset[:20400, :]
store_dataset(train, os.path.join(basic_path, "ALOI_train_20400.h5"))
test = dataset[20400:, :]
store_dataset(train, os.path.join(basic_path, "ALOI_test_3600.h5"))
hd.close()

# Google: We have already shuffled the images when aggregating the embedding files. Now we just split
file = "Google.h5"
hd = h5py.File(os.path.join(basic_path, file), 'r')
dataset = hd['dataset_1']
train = dataset[:6675, :]
store_dataset(train, os.path.join(basic_path, "Google_train_6675.h5"))
test = dataset[6675:, :]
store_dataset(train, os.path.join(basic_path, "Google_train_1178.h5"))
hd.close()

# ----------------------------------------------------------------------------------------------------------------------
"""This section splits the embeddings for different segments of images into test/train"""
for part in range(1,6):
    part_id = str(part)
    basic_path = "Datasets/" + layer_name + "/part_" + part_id

    # # Nexus dataset
    # file = "Nexus.h5"
    # hd = h5py.File(os.path.join(basic_path, file), 'r')
    # dataset = hd['dataset_1']
    # train = np.vstack((dataset[:800, :], dataset[1020:, :]))
    # store_dataset(train, os.path.join(basic_path, "part_" + part_id + "Nexus_train_1180.h5"))
    # test = dataset[801:1020, :]
    # store_dataset(train, os.path.join(basic_path, "part_" + part_id + "Nexus_test_220.h5"))
    # hd.close()

    # ALOI: We have already shuffled the images when aggregating the embedding files. Now we just split
    file = "ALOI.h5"
    hd = h5py.File(os.path.join(basic_path, file), 'r')
    dataset = hd['dataset_1']
    train = dataset[:20400, :]
    store_dataset(train, os.path.join(basic_path, "part_" + part_id + "ALOI_train_20400.h5"))
    test = dataset[20400:, :]
    store_dataset(train, os.path.join(basic_path, "part_" + part_id + "ALOI_test_3600.h5"))
    hd.close()

    # Google: We have already shuffled the images when aggregating the embedding files. Now we just split
    file = "Google.h5"
    hd = h5py.File(os.path.join(basic_path, file), 'r')
    dataset = hd['dataset_1']
    train = dataset[:6675, :]
    store_dataset(train, os.path.join(basic_path, "part_" + part_id + "Google_train_6675.h5"))
    test = dataset[6675:, :]
    store_dataset(train, os.path.join(basic_path, "part_" + part_id + "Google_train_1178.h5"))
    hd.close()













