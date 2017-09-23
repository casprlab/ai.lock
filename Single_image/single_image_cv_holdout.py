import numpy as np
import os
import errno
import sys
sys.path.append("/lclhome/mazim003/Documents/Projects/ai.lock/code") # the path to nearpy lib
from nearpy.hashes import RandomBinaryProjections
from nearpy import Engine
from nearpy import Engine
from Transform_PCA import TransformImagesPCA
import pandas as pd
from Test_case_attack_creator import Test_case_attack_creator
import h5py


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def project_LSH(dataset, rbp):
    data_transpose = np.transpose(dataset)
    data_hash = np.transpose(rbp.hash_vector(data_transpose, querying=True))
    return data_hash

def find_pcs(basic_path, layer_name):
    print("Finding PCs for layer: {}" .format(layer_name))
    basic_path_layer = os.path.join(basic_path, layer_name)
    dataset_files = "ALOI_train_20400.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_aloi = hd['dataset_1']
    transformer = TransformImagesPCA(n_components=500)
    transformer.learn_pcs(dataset_aloi)
    del dataset_aloi

    dataset_files = "Google_train_6675.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_google = hd['dataset_1']
    transformer.learn_pcs(dataset_google)
    del dataset_google

    dataset_files = "Nexus_train_1180.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset = hd['dataset_1']
    transformer.learn_pcs(dataset)
    del dataset
    return transformer

def data_for_experiment(basic_path, layer_names, projection_count, start_pc_component, end_pc_component, transformer):
    'For each layer in list of name of layers find PCs and LSH seperately and concatenate the results for multiple layers'
    print("layer {}".format(layer_names[0]))
    pc_test_nexus, pc_test_aloi, pc_test_google = test_data_for_layer(basic_path, layer_names[0],
                                                                 projection_count, start_pc_component, end_pc_component, transformer[0])
    for layer in range(1, len(layer_names)):
        print("layer {}".format(layer_names[layer]))
        pc_test_nexus1, pc_test_aloi1, pc_test_google1 = test_data_for_layer(basic_path, layer_names[layer],
                                                                             projection_count, start_pc_component, end_pc_component, transformer[layer])
        pc_test_nexus = np.column_stack((pc_test_nexus, pc_test_nexus1))
        pc_test_aloi = np.column_stack((pc_test_aloi, pc_test_aloi1))
        pc_test_google = np.column_stack((pc_test_google, pc_test_google1))
    return pc_test_nexus, pc_test_aloi, pc_test_google

def test_data_for_layer(basic_path, layer_name ,projection_count, start_pc_component, end_pc_component, transformer):
    # Read datasets
    basic_path_layer = os.path.join(basic_path, layer_name)
    dataset_files = "ALOI_test_3600.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files),'r')
    dataset_test_aloi = hd['dataset_1']

    pc_test_aloi = transformer.transform(dataset_test_aloi)[:, start_pc_component:end_pc_component]
    del dataset_test_aloi

    # Find the LSH vectors
    rbp = RandomBinaryProjections('rbp', projection_count, rand_seed=723657345)
    engine = Engine(end_pc_component - start_pc_component, lshashes=[rbp])

    pc_test_aloi = project_LSH(pc_test_aloi, rbp)

    dataset_files = "Google_test_1178.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_test_google = hd['dataset_1']
    pc_test_google = transformer.transform(dataset_test_google)[:, start_pc_component:end_pc_component]
    del dataset_test_google
    pc_test_google = project_LSH(pc_test_google, rbp)

    dataset_files = "Nexus_test_220.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_test_nexus = hd['dataset_1']
    pc_test_nexus = transformer.transform(dataset_test_nexus)[:, start_pc_component:end_pc_component]
    del dataset_test_nexus
    pc_test_nexus = project_LSH(pc_test_nexus, rbp)
    return pc_test_nexus,  pc_test_aloi,  pc_test_google


def perform_test(pc_nexus, pc_aloi, pc_google, out_path):
    # Train Threshold
    labels_nexus = []
    for index in range(0, pc_nexus.shape[0] // 4):
        labels_nexus.extend([index] * 4)

    overall_target = []
    overall_score = []

    pairs_train, labels_train = creat_all_pair_nexus(pc_nexus, labels_nexus,out_path)
    scores_list = find_scores(pairs_train)
    overall_target.extend(list(labels_train))
    overall_score.extend(scores_list)
    del pairs_train, labels_train

    attack_creator = Test_case_attack_creator(pc_nexus, pc_aloi)
    num_attack_samples_aloi = pc_nexus.shape[0] * pc_aloi.shape[0]
    for attack in range(num_attack_samples_aloi):
        pair_left, pair_right, labels_train = attack_creator.get_next_pair()
        pairs_train = np.array([[pair_left, pair_right]])
        scores_list = find_scores(pairs_train)
        overall_target.append(labels_train)
        overall_score.extend(scores_list)

    attack_creator = Test_case_attack_creator(pc_nexus, pc_google)
    num_attack_samples_google = pc_nexus.shape[0] * pc_google.shape[0]
    for attack in range(num_attack_samples_google):
        pair_left, pair_right, labels_train = attack_creator.get_next_pair()
        pairs_train = np.array([[pair_left, pair_right]])
        scores_list = find_scores(pairs_train)
        overall_target.append(labels_train)
        overall_score.extend(scores_list)
    print("overal attack dataset size: {}, {}" .format(len(overall_score), len(overall_target)))
    overall_score = np.array(overall_score)
    overall_target = np.array(overall_target)
    return overall_target, overall_score

def apply_threshold(threshold, test_pair_labels, scores, pos_label=None):
    # ensure binary classification if pos_label is not specified
    y_true = (test_pair_labels == pos_label)
    y_score = (scores >= threshold)
    positive = sum(map(bool, y_true))
    negative = len(y_true) - positive

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_score)):
        if y_score[i] and y_true[i]:
            tp+=1
        elif y_score[i] and not y_true[i]:
            fp+=1
        elif not y_score[i] and not y_true[i]:
            tn+=1
        else:
            fn+=1

    tpr = tp / float(positive)
    tnr = tn / float(negative)
    fpr = fp / float(negative)
    fnr = fn / float(positive)
    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    f_measure = (2 * tp) / float((2 * tp) + fn + fp)
    return tpr, fpr, fnr, tnr, accuracy, f_measure

def find_scores(test_pairs):
    def xor_hash(candidate_hash, ref_hash):
        return np.logical_xor(candidate_hash, ref_hash).tolist().count(1)
    scores = []
    length_vector = test_pairs[0, 0].shape[0]
    num_pairs = test_pairs.shape[0]
    for i in range(num_pairs):
        pair_right = test_pairs[i, 0]
        pair_lef = test_pairs[i, 1]
        dist_val = xor_hash(pair_lef, pair_right)
        sim = 1.0 - (dist_val/float(length_vector))
        if sim < 0 or sim > 1:
            print("error")
        scores.append(sim)
    return scores

def creat_all_pair_nexus(images, labels, path):
    pairs = []
    labels_out = []
    num_images = images.shape[0]
    reference_ids = []
    candidate_ids = []
    for i in range(num_images - 1):
        for j in range(i + 1, num_images):
            reference_ids.append(i)
            candidate_ids.append(j)
            pairs += [[images[i], images[j]]]
            if labels[i] == labels[j]:
                labels_out += [1]
            else:
                labels_out += [0]
    return np.array(pairs), np.array(labels_out)

def create_all_attack_pairs(X_test, y_test, X_attack, y_attack, flag_split = False, start = 0, end =24000):
    pairs = []
    labels_out = []
    num_images_1 = X_test.shape[0]
    num_images_attack = X_attack.shape[0]
    if not flag_split:
        for i in range(num_images_1):
            for j in range(num_images_attack):
                pairs += [[X_test[i], X_attack[j]]]
                if y_test[i] == y_attack[j]:
                    labels_out += [1]
                else:
                    labels_out += [0]
    else:
        for i in range(num_images_1):
            for j in range(start, end):
                pairs += [[X_test[i], X_attack[j]]]
                if y_test[i] == y_attack[j]:
                    labels_out += [1]
                else:
                    labels_out += [0]
    return np.array(pairs), np.array(labels_out)

def apply_threshold_fast(threshold, test_pair_labels, scores, pos_label=None):
    y_true = (test_pair_labels == pos_label)
    positive = sum(map(bool, y_true))
    negative = len(y_true) - positive
    desc_score_indices = np.argsort(scores, kind="mergesort")[::-1]
    y_score = scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    weight = 1.
    distinct_value_indices = []

    for th in threshold:
        a = np.where(y_score >= th)[0]
        if(a.size > 0):
            distinct_value_indices.append(a[-1])
        else:
            distinct_value_indices.append(-1)
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    cum_sum = (y_true * weight).cumsum()
    tps = [cum_sum[m] if m!=-1 else 0.0 for m in threshold_idxs]
    tps = np.array(tps)

    fps = 1 + threshold_idxs - tps
    fns = positive - tps
    tns = negative - fps

    accuracy = tps + tns + fps + fns
    accuracy = np.divide((tps + tns), accuracy)
    tp_2 = (2 * tps)
    f_measure = np.divide(tp_2, tp_2 + fps + fns)
    fps = fps / float(negative)
    tps = tps/float(positive)
    fns = fns/float(positive)
    tns = tns/float(negative)
    return tps[:-1], fps[:-1], fns[:-1], tns[:-1], accuracy[:-1], f_measure[:-1]

def main():
    """
    Computes the performance of MLSI or SLSI ai.lock on holdout set.
    to Switch between MLSI and SLSI modify the layer_names array parameter to specify the name of layers from inception
    v3 to be considered in imageprint calculation.
    """
    # the PC ranges that we want to try
    start_index_list = [0,  50, 0,   0,   0,   100, 0, 150, 200]
    n_components_list =[50, 50, 100, 150, 200, 100, 400, 150, 200]

    # length of imageprint (lambda)
    num_projections = range(50,501,50)

    thresholds_list = np.sort(np.concatenate((np.arange(0.0, 0.6, 0.01), np.arange(0.6, 1.0, 0.0001)), axis=0))

    basic_path = "Datasets"

    # layers to be considered. For SLSI only keep "bottleneck_FC" in the below list
    layer_names = ["Mixed8_pool0", "bottleneck_FC"]
    transformer = []
    for layer in range(len(layer_names)):
        transformer_pca = find_pcs(basic_path, layer_names[layer])
        transformer.append(transformer_pca)  # , layer_names[layer])

    for pc_index in range(len(start_index_list)):
        start_index = start_index_list[pc_index]
        n_components = n_components_list[pc_index]
        print("pc: {}-{}".format(start_index, start_index + n_components))

        training_data_path = "multilayer" + \
                             "/components = " + str(start_index) + "-" + str(start_index + n_components)

        # read the best threshold for binary classification from the training results
        train_data_info = pd.read_csv(os.path.join(training_data_path, "best_average_performance_test.txt"), sep="\t",
                                      header=None)
        train_data_info.drop(train_data_info.columns[0], axis=1, inplace=True)
        train_data_info = train_data_info.values
        best_threshold_index = np.argmax(train_data_info[:, train_data_info.shape[1]-1])
        thresholds = train_data_info[:, 1]

        for ind, n_proj in enumerate(num_projections):
            print("n_proj: {}" .format(n_proj))
            out_path = "multilayer_holdout" + "/components = " + str(start_index) + "-" + str(start_index + n_components)
            mkdir_p(out_path)
            pc_test_nexus, pc_test_aloi, pc_test_google = data_for_experiment(basic_path, layer_names, n_proj, start_index, start_index+n_components, transformer)
            overall_target, overall_score = perform_test(pc_test_nexus, pc_test_aloi, pc_test_google, out_path)
            tpr, fpr, fnr, tnr, accuracy, f_measure = apply_threshold(thresholds[ind], overall_target, overall_score,
                                                                      pos_label=1)
            with open(os.path.join(out_path, "performance.txt"), "a") as file:
                file.write(str(n_proj) + "\t" +
                           str(thresholds[ind]) + "\t" + "\t".join(map(str, [tpr, fpr, fnr, tnr, accuracy, f_measure])) + "\n")

            # # the following are optional: performance of best threshold of training, finding best threshold for holdout
            # # and sanity check for finding the best average performing threshold on training data
            # if ind == best_threshold_index:
            #     with open(os.path.join(out_path, "performance_on_best_training_thr.txt"), "a") as file:
            #         file.write(str(n_proj) + "\t" + str(thresholds[ind]) + "\t" + "\t".join(map(str, [tpr, fpr, fnr, tnr, accuracy, f_measure])) + "\n")
            #
            # # search best threshold for hold out data
            # tpr, fpr, fnr, tnr, accuracy, f_measure = apply_threshold_fast(thresholds_list, overall_target, overall_score,
            #                                                                pos_label=1)
            # arg_max = np.array(f_measure).argmax()
            # with open(os.path.join(out_path, "best_threshold_for_heldout.txt"), "a") as file:
            #     rec = [thresholds_list[arg_max], tpr[arg_max], fpr[arg_max], fnr[arg_max], accuracy[arg_max],
            #            f_measure[arg_max]]
            #     file.write(str(n_proj) + "\t" + "\t".join(map(str, rec)) + "\n")
            #
            # # sanity check
            # sanity_index = np.where(thresholds_list > thresholds[ind])[0][0] - 1
            # with open(os.path.join(out_path, "sanity_threshold_on_train.txt"), "a") as file:
            #     rec = [thresholds_list[sanity_index], tpr[sanity_index], fpr[sanity_index], fnr[sanity_index],
            #            accuracy[sanity_index],
            #            f_measure[sanity_index]]
            #     file.write(str(n_proj) + "\t" + "\t".join(map(str, rec)) + "\n")

if __name__ == "__main__":
    main()