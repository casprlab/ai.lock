import numpy

class Test_case_attack_creator:
    def __init__(self, test_dataset, dataset_attack):
        self.test_dataset = test_dataset
        self.dataset_attack = dataset_attack
        self.refid = 0
        self.canid = 0
        self.num_samples = self.test_dataset.shape[0]
        self.num_samples_attack = dataset_attack.shape[0]
    def get_next_pair(self):
        left_pair = self.test_dataset[self.refid,:]
        right_pair = self.dataset_attack[self.canid,:]
        label = 0
        self.canid += 1
        if self.canid == self.num_samples_attack:
            self.refid +=1
            self.canid = 0
        return left_pair, right_pair, label