import numpy as np
from random import choice, shuffle
from src.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import pandas as pd
nn=NeuralNetwork(784,10,hidden_layers=[16,16],learning_rate=2.5)
def main():
    data=pd.read_csv("data/train.csv")
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and 
    data_dev = data[0:1000].T#i use this because i have a lot of problems with my memory lol
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.
    x= X_dev[:, 32, None]
    y=Y_dev[32]
    nn.copy_from("nn.json")
    out=nn.prediction(x)
    x=x.reshape(28,28)
    plt.imshow(x,cmap='gray')
    plt.title("{} vs {}".format(y,out[0]))
   

    


    plt.show()
if __name__ == "__main__":  
    main()