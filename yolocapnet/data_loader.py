import numpy as np
import os

class DataLoader:
    def __init__(self, input_path=None):
        self.input_path = input_path

    def construct_train_file(self, output_path):
        if self.input_path is None: raise ValueError("Input path must be provided if a train file needs to be created")
        obj_file = open(os.path.join(self.input_path, 'obj.names'), 'r')
        classes = [line.rstrip().lstrip() for line in obj_file.readlines() if len(line) > 0]
        num_classes = len(classes)

        train_file = open(os.path.join(self.input_path, 'train.txt'), 'r')
        train_samples = [line.rstrip().lstrip().lstrip('data/').replace('png', 'txt') for line in train_file.readlines() if len(line) > 0]
        train_samples = [sample for sample in train_samples if os.path.exists(os.path.join(self.input_path, sample))]
        num_samples = len(train_samples)
        
        out_file = open(os.path.join(output_path, 'train_collated.txt'), 'w')
        sample_data = []
        for sample in train_samples:
            sample_file = open(os.path.join(self.input_path, sample), 'r')
            annotations = [line.rstrip().lstrip() for line in sample_file.readlines() if len(line) > 0]
            if len(annotations) == 0: continue
            args = [annotation.split(' ') for annotation in annotations]
            args = [[int(arg[0]), float(arg[1]), float(arg[2]), float(arg[3]), float(arg[4])] for arg in args]
            args = [[sample] + arg for arg in args][0]
            sample_data.append(args)
            out_file.write(" ".join(map(lambda x: str(x), args)) + '\n')
        out_file.close()

        return sample_data

    def load_train_file(self, train_file):
        if train_file is None: raise ValueError("The path to the train file that should be loaded must be provided")
        f = open(train_file, 'r')
        data = []
        for line in f.readlines():
            line = line.rstrip('\n')
            args = line.split(' ')
            data.append(args)
        
        return np.array(data)

if __name__ == '__main__': # test script
    train_path = 'data/'
    loader = DataLoader(input_path=train_path)

    loader.construct_train_file(train_path)

    # train_data = loader.load_train_file(train_path)

    # print(train_data)