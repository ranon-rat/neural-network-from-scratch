import numpy as np
from random import choice, shuffle
from src.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import pandas as pd
nn=NeuralNetwork(784,10,hidden_layers=[16,16],learning_rate=0.07)

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
    plt.imshow(x.reshape(28,28))
    plt.title("the actual number "+str(y) +" vs the predicted number "+str(out[0]))
   

    


    plt.show()
if __name__ == "__main__":  
    main()