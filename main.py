import numpy as np
from random import choice, shuffle
from src.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
testing=1
fig, ax = plt.subplots(1,1)
nen=NeuralNetwork(784,10,hidden_layers=[16,16],learning_rate=0.0007)
LEARNING_RATES = [0.025, 0.0025, 2.5]
def test_one(nen,target,input):
    
    
    print("|input:",input[:4],"...","target:",target)
    loss,output=nen.loss(target,input)
    print("predictions")
    for i in range(len(output)):
        print(i,":",output[i])
   
    print("|loss:",loss,"\n")
    print("-"*20)
def test(nen:NeuralNetwork,data):
    a=[]
    for i in data:
        target=[0]*10
        target[int(i[0])]= 1
        
        input=np.divide(i[1:],255)
      
        a.append({
            "target":target,
            "input":input
        })
    
    c=nen.gen_loss(a)
    return c

#    print("How exact it is?:",nen.exact(data))
def open_csv(path:str):
    data=np.loadtxt(path,delimiter=",")
    return data
def main():

    training_data=open_csv("data/mnist_train.csv")
    testing_data=open_csv("data/mnist_test.csv")
 
    

    c=[0]*30
  
    for i in range(30):
        shuffle(training_data)
        mini_batches = [
               training_data[k:k+10]
               for k in range(0, len(training_data), 10)]
        for m in mini_batches:
            for t in  m:
                target=[0]*10
                target[int(t[0])]= 1
                input=np.divide(t[1:],255)
                nen.train(target,input) 
        c[i]=test(nen,testing_data)
        print(c[i])
        
        
#
    
    plt.plot(c)
    for t in testing_data:
        target=[0]*10
        target[int(t[0])]= 1
        input=np.divide(t[1:],255)
        print("we want",t[0])
        test_one(nen,target,input)

    plt.show()
if __name__ == "__main__":
    main()