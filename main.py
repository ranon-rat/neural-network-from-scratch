from random import choice
from src.neural_network import NeuralNetwork
train=[
    {
        'input':[0,0],
        'target':[0]
    },
     {
        "input":[1,1],
        "target":[0]
    },
    {
        "input":[1,0],
        "target":[1]
    },
      {
        "input":[0,1],
        "target":[1]
    }, 
]
testing=1
def test(nen):
    global testing,train
    print(("-"*10),"TESTING #",testing,("-"*10))
    for t in train:
        print("\n","TESTING",t['input'],"TARGET",t['target'])
        print("\tinput:",t['input'],"target:",t['target'])
        cost,output=nen.cost(t['input'],t['target'])
        print("\tprediction:",output)
        print("\tcost:",cost)
    
   
   
    testing+=1
   
def main():

   
   
    nen=NeuralNetwork(2,1,[2])

  
    test(nen)
 
 
    for i in range(6000):
        t=choice(train)
        nen.train(t['input'],t['target'])
        nen.clear_nn()  
        
    test(nen)

 
if __name__ == "__main__":
    main()
