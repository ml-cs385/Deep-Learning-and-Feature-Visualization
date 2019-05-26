'''
This is the file for loading data
'''
# Necessary imports
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os

# Paths
# Declare your data paths
CIFAR_DATA_PATH = ""

'''
Father Class Dataloader
'''
class DataLoader:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        # Length of Train & Test Data
        self.train_length = None
        self.test_length = None

        self.train_batch_num = 0
        self.test_batch_num = 0
    
    def __save_to_pickle(self, data, name):
        fw = open(name + ".pkl", 'wb')
        pickle.dump(data, fw)
        fw.close()

    def save_dataset(self):
        self.__save_to_pickle(self.train_data, "train_data")
        self.__save_to_pickle(self.train_label, "train_label")
        self.__save_to_pickle(self.test_data, "test_data")
        self.__save_to_pickle(self.test_label, "test_label")

        print("Save pickle file successfully!")

    def load_pickle_dataset(self):
        fr_train_data = open("./train_data.pkl", 'rb')
        fr_train_label = open("./train_label.pkl", 'rb')
        fr_test_data = open("./test_data.pkl", 'rb')
        fr_test_label = open("./test_label.pkl", 'rb')

        self.train_data = pickle.load(fr_train_data)
        self.train_label = pickle.load(fr_train_label)
        self.test_data = pickle.load(fr_test_data)
        self.test_label = pickle.load(fr_test_label)
        
        fr_train_data.close()
        fr_train_label.close()
        fr_test_data.close()
        fr_test_label.close()

        print("========== Loading training and testing finished =========")

    def next_batch_train(self, batch_num=50):
        if (self.train_batch_num + batch_num > self.train_length):
            self.train_batch_num = 0
        batch_data, batch_label = self.train_data[self.train_batch_num: self.train_batch_num+batch_num], \
                                  self.train_label[self.train_batch_num: self.train_batch_num+batch_num]
        self.train_batch_num += batch_num
        return batch_data, batch_label

    def next_batch_test(self, batch_num=50):
        if (self.test_batch_num + batch_num > self.test_length):
            self.test_batch_num = 0
        batch_data, batch_label = self.test_data[self.test_batch_num: self.test_batch_num+batch_num], \
                                  self.test_label[self.test_batch_num: self.test_batch_num+batch_num]
        self.test_batch_num += batch_num
        return batch_data, batch_label

    '''
    In training neural network: to generate one hot vector
    '''
    def __one_hot(self, vec, vals=10): 
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out

class DataLoader_MNIST(DataLoader):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        mnist = tf.keras.datasets.mnist
        '''
        Notice: the mnist data loader in keras provide picture data as (?, 28, 28)
        I have expanded the dimension to (?, 28, 28, 1): have one channel
        All the image data will be 4 dimension in the following
        '''
        (self.train_data, self.train_label),(self.test_data, self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data, 3)
        self.test_data = np.expand_dims(self.test_data, 3)

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)

        print("Load Dataset Successfully! Train Data Length: %d\t Test Data Length: %d" %
            (self.train_length, self.test_length))

    def resize_images(self):
        pass

class DataLoader_CIFAR10(DataLoader):
    def __init__(self):
        super().__init__()

    def __unpickle(self, file):
        with open(os.path.join(CIFAR_DATA_PATH, file), 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def load_dataset(self):
        train_source = ["data_batch_{}".format(i) for i in range(1, 6)]
        test_source = ["test_batch"]
        # Loading training data
        data = [self.__unpickle(f) for f in train_source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.train_data = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.train_label = self.one_hot(np.hstack([d["labels"] for d in data]), 10)
        # Loading testing data
        data = [self.__unpickle(f) for f in test_source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.test_data = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.test_label = self.one_hot(np.hstack([d["labels"] for d in data]), 10)

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)

        print("Load Dataset Successfully! Train Data Length: %d\t Test Data Length: %d" %
            (self.train_length, self.test_length))

        

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
    for i in range(size)])
    plt.imshow(im)
    plt.show()  


if __name__ == "__main__":
    Data = DataLoader_MNIST()
    Data.load_dataset()
    Data.save_dataset()
    Data.load_pickle_dataset()
    print(Data.train_label.shape)

    data_X, data_y = Data.next_batch_train()
    print(data_X.shape)