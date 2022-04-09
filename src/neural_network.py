from html.entities import name2codepoint


from operator import index
from random import random
from re import T, template
from tkinter import W
import numpy as np 
#https://stackabuse.com/python-how-to-flatten-list-of-lists/
flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]
sigmoid= lambda x: 1/(1+np.exp(-x))
dsigmoid=np.vectorize(lambda y:y * (1 - y))


class NeuralNetwork():

    def __init__(self,length_of_input,length_of_output,hidden_layers=[2]):
        
        
       
        bias=[]
        weights=[]
        hidden_nodes=flatten_list(hidden_layers)
        
        layers=[length_of_input]+hidden_nodes+[length_of_output]
        nn=[[]]*len(layers)
        for l in range(len(layers)-1):
            bias.append(np.random.rand(layers[l+1],1))
            weights.append(np.random.rand(layers[l+1],layers[l]))
     
        bias[0]=np.zeros((length_of_input,1))
        self.nn=(nn)
        self.weights=(weights)
        self.bias=(bias)
        self.learning_rate=2

        

  
    

    def feed_foward(self,input):
       
        nn=self.nn
        
        
        weights=self.weights
        bias=self.bias
        nn[0]=np.matrix(input)
        for l in range(len(nn)-1):
            nn_t=np.reshape(nn[l],(-1,1))
            nn[l+1]=sigmoid(np.matmul(weights[l],nn_t)+bias[l])
        self.nn=nn  

        return nn[-1]



      
    def clear_nn(self)->None:
        for l in range(len(self.nn)):

            self.nn[l]=np.zeros(len(self.nn[l]))
   
  
    def cost(self,input,target):
        
        output=self.predict(input)
        return np.sum(np.square(output-target))
    def predict(self,input):
        output=self.feed_foward(input)
        self.clear_nn()
        return output   
    def train(self,input,target):
        
        output=self.feed_foward(input)
       
        errors=np.matrix(np.subtract(target,output))
     
        gradient=(dsigmoid(output))
        gradient=np.multiply(gradient,errors)
        gradient=np.multiply(gradient,self.learning_rate)
        l=len(self.nn)-2
        while l>=0:

            nn_t=np.transpose(self.nn[l])
            #print("pos",l)
            deltGrad= np.multiply( gradient,nn_t)
            #print("weights before",l,self.weights[l])
            self.weights[l]=np.add(self.weights[l], deltGrad)
           # print("weights after",l,self.weights[l])
          
            self.bias[l]=np.add(self.bias[l],gradient)
            if l==0:break   
            weight_t=np.transpose(self.weights[l])
            errors=np.matmul(weight_t,errors)
            gradient=dsigmoid(self.nn[l])
            gradient=np.multiply(gradient,errors)
            gradient=np.multiply(gradient,self.learning_rate)
            l-=1
        self.clear_nn()

      
