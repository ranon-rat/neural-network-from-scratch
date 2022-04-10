import numpy as np 
from json import dumps,loads
from time import time
#https://stackabuse.com/python-how-to-flatten-list-of-lists/



tanh={
    "activation":np.vectorize(lambda x:np.tanh(x)),
    "derivate":np.vectorize(lambda y:1-(y**2))
}
sigmoid={
    "activation":np.vectorize(lambda x:1/(1+np.exp(-x))),
    "derivate":np.vectorize(lambda y:y * (1 - y))
}
    
relu={
    "activation":np.vectorize(lambda x:max(0,x)),
    "derivate":np.vectorize(lambda y:np.where(y>0,1,0))
}
activation_functions={
    "tanh":tanh,
    "relu":relu,
    "sigmoid":sigmoid

}


class NeuralNetwork():

    def __init__(self,length_of_input:int,length_of_output:int,hidden_layers=[2], min_lr=1e-5,max_lr=1e-2,decay_factor=0.95,step_size=10,cycle_size=10):
        
        layers_length=[length_of_input]+hidden_layers+[length_of_output]
        bias=[]
        weights=[]       
        
        funcs=[relu]*len(hidden_layers)+[tanh]       
        layers=[[]]*len(layers_length)
        
        for l in range(len(layers_length)-1):
            bias.append(np.matrix(np.random.rand(layers_length[l+1],1)))
            weights.append(np.matrix(np.random.rand(layers_length[l+1],layers_length[l])))

        self.activation_funcs_names=["relu"]*len(hidden_layers)+["tanh"]
        self.activation_funcs=funcs
        
        self.layers_length=len(layers_length)
        self.layers=layers
        
        self.weights=(weights)
        self.bias=(bias)
        
        self.min_lr=min_lr
        self.max_lr=max_lr
        self.learning_rate=max_lr
        self.decay_factor=decay_factor

        self.step_size=step_size
        self.batch_size=0
        self.next_restart=cycle_size
    def copy_structure(self,nn:dict)->None:
        self.max_lr=nn["max_learning_rate"]
        self.min_lr=nn["min_learning_rate"]
        
        self.activation_funcs_names=nn["activation_funcs"]
        #im just using the names to access the functions
        self.activation_funcs=list(map(lambda x:activation_functions[x],nn["activation_funcs"]))
        # i make the same thing that i make when i define the class
        self.layers_length=nn["layers_amount"]
        self.layers=[[]]*nn["layers_amount"]
        # i just copy the weights and bias
        self.weights=list(map(np.matrix,nn["weights"]))
        self.bias=list(map(np.matrix,nn["bias"]))
    
    def give_structure(self)->dict:
        return {  
            "max_learning_rate":self.max_lr,
            "min_learning_rate":self.min_lr,
            "activation_funcs":self.activation_funcs_names,
            "layers_amount":self.layers_length,
            "weights":list(map(lambda x:x.tolist(), self.weights)),
            "bias":list(map(lambda x:x.tolist(), self.bias)),
           
        }
    def copy_from(self,path:str):
        with open(path, "r") as infile:
            nn=loads(infile.read())
            self.copy_structure((nn))
        
    def save(self,path:str="",name="neural_network"+str(time())+".json")->None:
        with open(path+name, "w") as outfile:
            s=dumps(self.give_structure(),indent = 4)
            outfile.write(s)

    def feed_foward(self,input:list)->np.matrix:
        layers=self.layers 
        weights=self.weights
        bias=self.bias
        layers[0]=np.matrix(input).transpose()

        for l in range(len(layers)-1):
            
            """
            http://matrixmultiplication.xyz/
            """
        
            res=np.matmul(weights[l],layers[l])
            layers[l+1]=self.activation_funcs[l]["activation"](res+bias[l])
        
        self.layers=layers
        return layers[-1]



      
    
    def general_cost(self,train:list)->float:
        # this is for getting the general cost
        # its ugly
        cost=sum(map(lambda x:self.cost(x["input"],x["target"])[0] ,train))
        return cost/len(train)
    def cost(self,input:float,target:float)->float:

        output=self.predict(input)
        return (np.sum(np.square(output-target))/len(target),output)

    def predict(self,input)->float:

        output=self.feed_foward(input)
        return output   
    #https://www.jeremyjordan.me/nn-learning-rate/
    def update_learning_rate(self)->None:
        x = self.batch_size/(self.step_size*self.next_restart)
        self.learning_rate=self.min_lr+0.5*(self.max_lr-self.min_lr)*(1+np.cos(x*np.pi))
 
    

    def train(self,iteration,input:float,target:float,)->None:
        # this is for updating the learning rate
        # with this the probability of getting stuck in local minima
        # is reduced
        self.update_learning_rate()

        self.backprop(input,target)
        self.batch_size+=1


        if iteration%self.step_size==0:
            self.batch_size=0
            self.next_restart+=self.step_size
            #self.max_lr*=self.decay_factor

    def gradient_descent(self,output:np.matrix,errors:np.matrix,derivate_func)->np.matrix:
        return np.multiply( np.multiply(errors,derivate_func(output)),self.learning_rate)
       
    def backprop(self,input:float,target:float):
        
        output=self.feed_foward(input)
        errors=np.subtract(target,output)
        gradient=self.gradient_descent(output,errors,self.activation_funcs[-1]["derivate"])
       
        for l in range(len(self.layers)-1)[::-1]:
          
            layers_t=np.transpose(self.layers[l])
            deltgrad=np.matmul(gradient,layers_t)
            # this is for getting the weights.. you know what i mean  
             # its the delta of the gradient so 
            # it will reduce the errors
            
            self.weights[l]=np.add(self.weights[l],deltgrad)
            self.bias[l]=np.add(self.bias[l],gradient)
            
            if l==0:break    

            weight_t=(self.weights[l]).transpose()
            errors=np.matmul(weight_t,errors)
            gradient=self.gradient_descent(self.layers[l],errors,self.activation_funcs[l-1]["derivate"])
        


      
