# Multiple image version of ai.lock.
import numpy as np
import os
import errno
import sys
sys.path.append("/lclhome/mazim003/Documents/Projects/ai.lock/code") # the path to nearpy lib
from nearpy.hashes import RandomBinaryProjections
from nearpy import Engine
from Transform_PCA import TransformImagesPCA
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


# we assume the dataset can equally be split into the num_folds folds
def split_data_to_test_train(dataset, num_folds, index_train_fold):
    """
    Split data to test and train folds for cross validation
    """
    fold_size = (dataset.shape[0] / num_folds)
    start_index_for_test_fold = index_train_fold * fold_size
    end_index_for_test_fold = (index_train_fold + 1) * fold_size
    test_indices = range(start_index_for_test_fold, end_index_for_test_fold)
    mask = np.ones(dataset.shape[0], dtype=bool)
    mask[test_indices] = False
    train_indices = list(np.where(mask)[0])
    test_set = dataset[test_indices, :]
    train_set = dataset[train_indices, :]
    return train_set, test_set


def project_LSH(dataset, rbp):
    """
    Compute the imageprints (LSH projected vector)
    """
    data_transpose = np.transpose(dataset)
    data_hash = np.transpose(rbp.hash_vector(data_transpose, querying=True))
    return data_hash


def test_data_for_cv_experiment(basic_path, layer_names, num_folds, experiment, projection_count, start_pc_component,
                                end_pc_component, image_part):
    """
    Use test_data_cv_for_layer method for each layer in the list of name of layers separately to find PCs and LSH vectors.
    Then concatenate the results of each layer to create multi-layer imageprint.
    """
    # For each layer in list of name of layers find PCs and LSH seperately and concatenate the results for multiple layers
    # print("layer {}".format(layer_names[0]))
    pc_test_nexus, pc_test_aloi, pc_test_google = test_data_cv_for_layer(basic_path, layer_names[0], num_folds,
                                                                         experiment,
                                                                         projection_count, start_pc_component,
                                                                         end_pc_component, image_part)
    for layer in range(1, len(layer_names)):
        pc_test_nexus1, pc_test_aloi1, pc_test_google1 = test_data_cv_for_layer(basic_path, layer_names[layer],
                                                                                num_folds, experiment,
                                                                                projection_count, start_pc_component,
                                                                                end_pc_component, image_part)
        pc_test_nexus = np.column_stack((pc_test_nexus, pc_test_nexus1))
        pc_test_aloi = np.column_stack((pc_test_aloi, pc_test_aloi1))
        pc_test_google = np.column_stack((pc_test_google, pc_test_google1))
    print("test fold shapes: {}, {}, {}".format(pc_test_nexus.shape, pc_test_aloi.shape, pc_test_google.shape))
    return pc_test_nexus, pc_test_aloi, pc_test_google


def test_data_cv_for_layer(basic_path, layer_name, num_folds, experiment, projection_count, start_pc_component,
                           end_pc_component, image_part):
    # Read datasets
    basic_path_layer = os.path.join(basic_path, layer_name)
    dataset_files = "ALOI_train_20400.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_aloi = hd['dataset_1']
    dataset_train_aloi, _ = split_data_to_test_train(dataset_aloi, num_folds, experiment)
    del dataset_aloi
    transformer = TransformImagesPCA(n_components=500)
    transformer.learn_pcs(dataset_train_aloi)
    del dataset_train_aloi

    dataset_files = "Google_train_6675.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset_google = hd['dataset_1']
    dataset_train_google, _ = split_data_to_test_train(dataset_google, num_folds, experiment)
    del dataset_google
    transformer.learn_pcs(dataset_train_google)
    del dataset_train_google

    dataset_files = "Nexus_train_1180.h5"
    hd = h5py.File(os.path.join(basic_path_layer, "full_size", dataset_files), 'r')
    dataset = hd['dataset_1']
    dataset_train, _ = split_data_to_test_train(dataset, num_folds, experiment)
    del dataset
    transformer.learn_pcs(dataset_train)
    del dataset_train

    # The data for test is not from the complete image but the parts
    dataset_files = "ALOI_train_20400.h5"
    hd = h5py.File(
        os.path.join(basic_path_layer, "part_" + str(image_part), "part_" + str(image_part) + "_" + dataset_files), 'r')
    dataset_aloi = hd['dataset_1']
    _, dataset_test_aloi = split_data_to_test_train(dataset_aloi, num_folds, experiment)
    del dataset_aloi

    dataset_files = "Google_train_6675.h5"
    hd = h5py.File(
        os.path.join(basic_path_layer, "part_" + str(image_part), "part_" + str(image_part) + "_" + dataset_files), 'r')
    dataset_google = hd['dataset_1']
    _, dataset_test_google = split_data_to_test_train(dataset_google, num_folds, experiment)
    del dataset_google

    dataset_files = "Nexus_train_1180.h5"
    hd = h5py.File(
        os.path.join(basic_path_layer, "part_" + str(image_part), "part_" + str(image_part) + "_" + dataset_files), 'r')
    dataset = hd['dataset_1']
    _, dataset_test = split_data_to_test_train(dataset, num_folds, experiment)
    del dataset

    # calculate PCs based on half of training data (only nexus)
    pc_test_nexus = transformer.transform(dataset_test)[:, start_pc_component:end_pc_component]
    pc_test_aloi = transformer.transform(dataset_test_aloi)[:, start_pc_component:end_pc_component]
    pc_test_google = transformer.transform(dataset_test_google)[:, start_pc_component:end_pc_component]

    # Find the LSH vectors
    rbp = RandomBinaryProjections('rbp', projection_count, rand_seed=723657345)
    engine = Engine(end_pc_component - start_pc_component, lshashes=[rbp])

    pc_test_nexus = project_LSH(pc_test_nexus, rbp)
    pc_test_aloi = project_LSH(pc_test_aloi, rbp)
    pc_test_google = project_LSH(pc_test_google, rbp)
    return pc_test_nexus, pc_test_aloi, pc_test_google


def find_scores_attack(dataset_test, dataset_attack):
    def xor_hash(candidate_hash, ref_hash):
        return np.logical_xor(candidate_hash, ref_hash).tolist().count(1)

    scores = []
    labels = []
    length_vector = dataset_test[0].shape[0]
    num_pairs = dataset_test.shape[0] * dataset_attack.shape[0]
    test_case = Test_case_attack_creator(dataset_test, dataset_attack)
    for i in range(num_pairs):
        pair_lef, pair_right, label = test_case.get_next_pair()
        dist_val = xor_hash(pair_lef, pair_right)
        sim = 1.0 - (dist_val / float(length_vector))
        if sim < 0 or sim > 1:
            print("error")
        scores.append(sim)
        labels.append(label)
    return scores, labels


def perform_test(pc_nexus, pc_aloi, pc_google):
    """
    Run the test for a single fold
    """
    # Train Threshold
    labels_nexus = []
    for index in range(0, pc_nexus.shape[0] // 4):
        labels_nexus.extend([index] * 4)

    overall_target = []
    overall_score = []

    pairs_train, labels_train = creat_all_pair_nexus(pc_nexus, labels_nexus)
    scores_list = find_scores(pairs_train)
    overall_target.extend(list(labels_train))
    overall_score.extend(scores_list)
    del pairs_train, labels_train

    # ALOI Training
    scores_list, labels_train = find_scores_attack(pc_nexus, pc_aloi)
    overall_target.extend(labels_train)
    overall_score.extend(scores_list)

    # Google
    scores_list, labels_train = find_scores_attack(pc_nexus, pc_google)
    overall_target.extend(labels_train)
    overall_score.extend(scores_list)

    overall_score = np.array(overall_score)
    overall_target = np.array(overall_target)
    print("overall size test: {}, +: {}, -: {}".format(overall_score.shape, (overall_target == 1).sum(),
                                                       (overall_target == 0).sum()))
    return np.array(overall_target), np.array(overall_score)


def apply_threshold(threshold, test_pair_labels, scores, pos_label=None):
    """
    Compute the classifier performance given a threshold, actual and predicted scores
    """
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
            tp += 1
        elif y_score[i] and not y_true[i]:
            fp += 1
        elif not y_score[i] and not y_true[i]:
            tn += 1
        else:
            fn += 1
    tpr = tp / float(positive)
    tnr = tn / float(negative)
    fpr = fp / float(negative)
    fnr = fn / float(positive)
    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    f_measure = (2 * tp) / float((2 * tp) + fn + fp)
    return tpr, fpr, fnr, tnr, accuracy, f_measure


def train(basic_path, layer_names, out_path, num_folds, experiment, projection_count, start_pc_component,
          end_pc_component, thresholds, image_part):
    """
    Finds the best performing threshold for binary classification
    """
    # gives us the data for one specific part (but LSH of different layers are concatenated
    pc_test_nexus, pc_test_aloi, pc_test_google = \
        test_data_for_cv_experiment(basic_path, layer_names, num_folds, experiment, projection_count,
                                    start_pc_component, end_pc_component, image_part)
    # on  Test
    overall_target, overall_score = perform_test(pc_test_nexus, pc_test_aloi, pc_test_google)
    tpr, fpr, fnr, tnr, accuracy, f_measure = apply_threshold_fast(thresholds, overall_target, overall_score,
                                                                   pos_label=1.0)
    arg_max = f_measure.argmax()
    best_threshold = thresholds[arg_max]
    with open(os.path.join(out_path, "fold_test_performance_part_" + str(image_part) + ".txt"), 'a') as file:
        ress = [thresholds[arg_max], tpr[arg_max], fpr[arg_max], fnr[arg_max], tnr[arg_max], accuracy[arg_max],
                f_measure[arg_max]]
        rec = "\t".join(map(str, ress))
        file.write(rec + "\n")
    res = np.array([tpr, fpr, fnr, tnr, accuracy, f_measure])
    return best_threshold, res, overall_score, overall_target


def find_scores(test_pairs):
    """
    Finds the matching score between imageprint pairs
    """
    def xor_hash(candidate_hash, ref_hash):
        candidate_code = candidate_hash.tolist()
        ref_code = ref_hash.tolist()
        xored = [0 if a == b else 1 for a, b in zip(candidate_code, ref_code)]
        return xored.count(1)

    scores = []
    length_vector = test_pairs[0, 0].shape[0]
    num_pairs = test_pairs.shape[0]
    for i in range(num_pairs):
        pair_right = test_pairs[i, 0]
        pair_lef = test_pairs[i, 1]
        dist_val = xor_hash(pair_lef, pair_right)
        sim = 1.0 - (dist_val / float(length_vector))
        if sim < 0 or sim > 1:
            print("error")
        scores.append(sim)
    return scores


def creat_all_pair_nexus(images, labels):
    """
    Creates all the possible image pairs for images in Nexus dayaset
    """
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


def create_all_attack_pairs(X_test, y_test, X_attack, y_attack, flag_split=False, start=0, end=24000):
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
        if (a.size > 0):
            distinct_value_indices.append(a[-1])
        else:
            distinct_value_indices.append(-1)
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    cum_sum = (y_true * weight).cumsum()
    tps = [cum_sum[m] if m != -1 else 0.0 for m in threshold_idxs]
    tps = np.array(tps)

    fps = 1 + threshold_idxs - tps
    fns = positive - tps
    tns = negative - fps

    accuracy = tps + tns + fps + fns
    accuracy = np.divide((tps + tns), accuracy)
    tp_2 = (2 * tps)
    f_measure = np.divide(tp_2, tp_2 + fps + fns)
    fps = fps / float(negative)
    tps = tps / float(positive)
    fns = fns / float(positive)
    tns = tns / float(negative)
    return tps[:-1], fps[:-1], fns[:-1], tns[:-1], accuracy[:-1], f_measure[:-1]


def find_pcs(basic_path, layer_name):
    # for each split of the image the PCs are found by looking at the complete size images
    print("Finding PCs for layer: {}".format(layer_name))
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


def data_for_experiment_heldout(basic_path, layer_names, projection_count, start_pc_component, end_pc_component,
                                transformer, image_part):
    # For each layer in list of name of layers find PCs and LSH seperately and concatenate the results for multiple layers
    pc_test_nexus, pc_test_aloi, pc_test_google = heldout_data_for_layer(basic_path, layer_names[0],
                                                                         projection_count, start_pc_component,
                                                                         end_pc_component, transformer[0], image_part)
    for layer in range(1, len(layer_names)):
        pc_test_nexus1, pc_test_aloi1, pc_test_google1 = heldout_data_for_layer(basic_path, layer_names[layer],
                                                                                projection_count, start_pc_component,
                                                                                end_pc_component, transformer[layer],
                                                                                image_part)
        pc_test_nexus = np.column_stack((pc_test_nexus, pc_test_nexus1))
        pc_test_aloi = np.column_stack((pc_test_aloi, pc_test_aloi1))
        pc_test_google = np.column_stack((pc_test_google, pc_test_google1))
        print(
        "dataset shapes heldout: {}, {}, {}".format(pc_test_nexus.shape, pc_test_aloi.shape, pc_test_google.shape))
    return pc_test_nexus, pc_test_aloi, pc_test_google


def heldout_data_for_layer(basic_path, layer_name, projection_count, start_pc_component, end_pc_component, transformer,
                           image_part):
    # Read datasets
    basic_path_layer = os.path.join(basic_path, layer_name)
    dataset_files = "ALOI_test_3600.h5"
    hd = h5py.File(
        os.path.join(basic_path_layer, "part_" + str(image_part), "part_" + str(image_part) + "_" + dataset_files), 'r')
    dataset_test_aloi = hd['dataset_1']

    pc_test_aloi = transformer.transform(dataset_test_aloi)[:, start_pc_component:end_pc_component]
    del dataset_test_aloi

    # Find the LSH vectors
    rbp = RandomBinaryProjections('rbp', projection_count, rand_seed=723657345)
    engine = Engine(end_pc_component - start_pc_component, lshashes=[rbp])

    pc_test_aloi = project_LSH(pc_test_aloi, rbp)

    dataset_files = "Google_test_1178.h5"
    hd = h5py.File(
        os.path.join(basic_path_layer, "part_" + str(image_part), "part_" + str(image_part) + "_" + dataset_files), 'r')
    dataset_test_google = hd['dataset_1']
    pc_test_google = transformer.transform(dataset_test_google)[:, start_pc_component:end_pc_component]
    del dataset_test_google
    pc_test_google = project_LSH(pc_test_google, rbp)

    dataset_files = "Nexus_test_220.h5"
    hd = h5py.File(
        os.path.join(basic_path_layer, "part_" + str(image_part), "part_" + str(image_part) + "_" + dataset_files), 'r')
    dataset_test_nexus = hd['dataset_1']
    pc_test_nexus = transformer.transform(dataset_test_nexus)[:, start_pc_component:end_pc_component]
    del dataset_test_nexus
    pc_test_nexus = project_LSH(pc_test_nexus, rbp)
    print("dataset shapes heldout: {}, {}, {}".format(pc_test_nexus.shape, pc_test_aloi.shape, pc_test_google.shape))
    return pc_test_nexus, pc_test_aloi, pc_test_google


def main():
    # PC ranges to be tried
    start_index_list = [0, 0, 0, 100, 150, 200, 0]
    n_components_list = [100, 150, 200, 100, 150, 200, 400]

    # the values to try for lambda
    num_projections = range(50, 501, 100)

    # root directory for datasets
    basic_path = "Datasets"

    # the name of layers that you want to consider. For SLMI only keep bottleneck_fc from the list of layers
    layer_names = ["Mixed8_Pool0", "bottleneck_FC"]

    # number of image segments
    image_parts = range(1, 6)

    transformer_heldout = []
    # finding the PCA transformer based on all the training data
    for layer in range(len(layer_names)):
        transformer_pca = find_pcs(basic_path, layer_names[layer])
        transformer_heldout.append(transformer_pca)

    thresholds = np.sort(np.concatenate((np.arange(0.0, 0.6, 0.01), np.arange(0.6, 1.0, 0.0001)), axis=0))

    for pc_index in range(len(start_index_list)):
        start_index = start_index_list[pc_index]
        n_components = n_components_list[pc_index]
        print("pc: {}-{}".format(start_index, start_index + n_components))
        for n_proj in num_projections:
            print("n_proj: {}".format(n_proj))
            num_folds = 5
            list_of_best_th_for_parts = []
            overall_score_parts = []

            for image_part in image_parts:
                print("image part: {}".format(image_part))
                list_performance_test = []
                best_threshold_sum = 0.0
                overall_score_list_folds = []
                out_path = "CV_train_MLMI" + \
                           "/components = " + str(start_index) + "-" + str(
                    start_index + n_components) + "/n_proj=" + str(n_proj)
                mkdir_p(out_path)
                for experiment in range(0, num_folds):
                    print("fold: {}".format(experiment + 1))
                    best_threshold, perf, overall_score, overall_target = train(basic_path, layer_names, out_path,
                                                                                num_folds, experiment, n_proj,
                                                                                start_index, start_index + n_components,
                                                                                thresholds, image_part)
                    # take the average of best performing ths in folds
                    best_threshold_sum += best_threshold
                    list_performance_test.append(perf)
                    overall_score_list_folds.append(overall_score)

                test_threshold = best_threshold_sum / float(num_folds)
                list_performance_test = sum(list_performance_test) / num_folds
                last_index = list_performance_test.shape[0] - 1
                performance_f1 = list_performance_test[last_index]
                arg_max = np.argmax(performance_f1)
                performance_test = list_performance_test[:, arg_max]
                list_of_best_th_for_parts.append(thresholds[arg_max])
                with open(os.path.join(out_path, "best_average_performance_test_part_" + str(image_part) + ".txt"),
                          "a") as file:
                    performance_test = list(performance_test)
                    rec = "\t".join(map(str, performance_test))
                    record = str(start_index) + "-" + str(start_index + n_components) + "\t" + str(n_proj) + "\t" + str(
                        thresholds[arg_max]) + "\t" \
                             + rec
                    file.write(record + "\n")
                with open(os.path.join(out_path, "average_bestperformingth_test_part_" + str(image_part) + ".txt"),
                          'a') as file:
                    file.write(str(test_threshold) + "\n")
                with open(os.path.join(out_path, "list_thresholds_for_part_" + str(image_part) + ".txt"), 'a') as file:
                    file.write(str(list_of_best_th_for_parts[-1]))
                overall_score_list_folds = np.array(overall_score_list_folds)
                overall_score_list_folds = [overall_score_list_folds >= thresholds[arg_max]]
                overall_score_list_folds = sum(overall_score_list_folds)
                overall_score_parts.append(overall_score_list_folds)

            # ----------------------------------------------------------------------------------------------------------
            # find the performance on training based on the final matching threshold (3,4 or 5)
            overall_score_test_folds = sum(overall_score_parts)
            for num_match in range(2, max(image_parts) + 1):
                list_performance_test_folds = []
                for experiment in range(0, num_folds):
                    tpr, fpr, fnr, tnr, accuracy, f_measure = apply_threshold(num_match, overall_target,
                                                                              overall_score_test_folds[experiment],
                                                                              pos_label=1)
                    perf = np.array([tpr, fpr, fnr, tnr, accuracy, f_measure])
                    list_performance_test_folds.append(perf)
                list_performance_test_folds = sum(list_performance_test_folds) / num_folds
                with open(
                        os.path.join(os.path.dirname(out_path),
                                     "avg_performance_test_folds_numMatch_" + str(num_match) + ".txt"),
                        "a") as file:
                    rec = "\t".join(map(str, list_performance_test_folds))
                    file.write(
                        str(start_index) + "-" + str(start_index + n_components) + "\t" + str(
                            n_proj) + "\t" + rec + "\n")

            # ----------------------------------------------------------------------------------------------------------
            # perform test on heldout
            list_binary_scores = []
            for image_part in image_parts:
                pc_test_nexus, pc_test_aloi, pc_test_google = data_for_experiment_heldout(basic_path, layer_names,
                                                                                          n_proj,
                                                                                          start_index,
                                                                                          start_index + n_components,
                                                                                          transformer_heldout,
                                                                                          image_part)
                overall_target, overall_score = perform_test(pc_test_nexus, pc_test_aloi, pc_test_google)
                y_score = (overall_score >= list_of_best_th_for_parts[image_part - 1])
                list_binary_scores.append(y_score)
                del overall_score

            overall_test_matching_counter = sum(list_binary_scores)
            for num_match in range(2, max(image_parts) + 1):
                tpr, fpr, fnr, tnr, accuracy, f_measure = apply_threshold(num_match, overall_target,
                                                                          overall_test_matching_counter,
                                                                          pos_label=1)
                with open(os.path.join(os.path.dirname(out_path),
                                       "performance_heldout_numMatch_" + str(num_match) + ".txt"), "a") as file:
                    rec = "\t".join(map(str, [tpr, fpr, fnr, tnr, accuracy, f_measure]))
                    file.write(str(start_index) + "-" + str(start_index + n_components) + "\t" + str(
                        n_proj) + "\t" + rec + "\n")


if __name__ == "__main__":
    main()
