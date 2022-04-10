from src.neural_network import NeuralNetwork
from json import loads
nen=NeuralNetwork(1,1,[0])


testing=1


def test(nen,data):
    global testing
    print(("#"*10),"TESTING #",testing,("#"*10))
    for t in data:
        print("\nTESTING",t['input'],"TARGET",t['target'])
        print("\ninput:",t['input'],"target:",t['target'])
        cost,output=nen.cost(t['input'],t['target'])
        print("prediction:",output)
        print("cost:",cost,"\n")
        print("-"*20)
    
    testing+=1
    print("GENERAL COST:",nen.general_cost(data))

def main():

    data=loads(open("data.json").read())
    nen.copy_from("nn.json")

    test(nen,data)
    
if __name__ == "__main__":
    main()
