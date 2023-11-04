import numpy as np
import os

class DataLoader:
    def __init__(self, input_path=None):
        self.input_path = input_path

    def construct_train_file(self, output_path):
        if self.input_path is None: raise ValueError("Input path must be provided if a train file needs to be created")
        pass

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
    loader = DataLoader()
    train_path = 'data/train.txt'

    train_data = loader.load_train_file(train_path)

    print(train_data)