from random import choice
from src.neural_network import NeuralNetwork

train=[
    {
        'input':[0,0],
        'target':[0]
    },
    {
        "input":[0,1],
        "target":[1]
    },
    {
        "input":[1,0],
        "target":[1]

    },
    {
        "input":[1,0],
        "target":[1]

    },
    {
        "input":[1,1],
        "target":[0]
    },
     {
        'input':[0,0],
        'target':[0]
    },
    {
        "input":[0,1],
        "target":[1]
    },
 
    {
        "input":[1,1],
        "target":[0]
    }
]
def main():
   
   
    nen=NeuralNetwork(2,1,[2,4])

  
    print(nen.feed_foward([0,0]))
    print(nen.weights)
    for i in range(10000):
        t=choice(train)
        nen.train(t['input'],t['target'])
        nen.clear_nn()
       
        
    nen.clear_nn()
    print(nen.nn,nen.weights)
    nen.clear_nn()
    print("it should be near to 1     ",nen.feed_foward([1,0]))
    nen.clear_nn()
    print("the same as before,1       ",nen.feed_foward([0,1]))
    nen.clear_nn()
    print("it should be near to 0     ",nen.feed_foward([1,1]))
    nen.clear_nn()
    print("the same as before, nothing",nen.feed_foward([0,0]))
 
if __name__ == "__main__":
    main()
