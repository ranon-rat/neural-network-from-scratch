from math import exp

from operator import index
from random import random
from re import template

#https://stackabuse.com/python-how-to-flatten-list-of-lists/
flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]
sigmoid=lambda x: 1/(1+exp(-x))
dsigmoid=lambda y:y * (1 - y)
        

class NeuralNetwork():

    def __init__(self,length_of_input,length_of_output,hidden_layers=[2]):
        nn=[]
        hidden_nodes=flatten_list(hidden_layers)
        template_nn=[]
        weights=[]    
        bias=[]
        for length in [length_of_input]+hidden_nodes+[length_of_output]:
            nn.append([])
            
            bias.append([random()]*length)
            template_nn.append([])
            template_nn[-1]=[0]*length
            nn[-1]=[0]*length

            


        for l in range(len(nn)-1):
            weights.append([])

            for _ in range(len(nn[l])):
                weights[-1].append([random()]*len(nn[l+1]))

        bias[0]=0 
        
        self.nn=nn
        self.weights=weights
        self.bias=bias
        self.learning_rate=1
        

  
    

    def feed_foward(self,input):
    
        nn=self.nn
        if len(nn[0])!=len(input):
            raise ValueError("Inputs and NN inputs are not the same length")
        weights=self.weights
        bias=self.bias
        
        nn[0]=input
        for l in range(len(nn)):
            for n in range(len(nn[l])):
                if l!=0:     
                  
                    nn[l][n]+=bias[l][n]
                   
                    nn[l][n]=sigmoid(nn[l][n])
                if l==len(nn)-1:continue
                for w in range(len(weights[l][n])):
                    
                    nn[l+1][w]+=nn[l][n]*weights[l][n][w]
           

        output=nn[-1]
       
        return output



      
    def clear_nn(self)->None:
        for l in range(len(self.nn)):

            self.nn[l]=[0]*len(self.nn[l])
    def transpose(self,matrix):
        if type(matrix[0]) is not list:
            matrix=list(map(lambda x:[]+[x],matrix))
        out = list(map(list,zip(*matrix)))     
        return out
    def multiplyOne(self,a,b):
       

        if type(a[0]) !=  list:
            a=list(map(lambda x:[]+[x],a))
        if type(b[0]) !=  list:
            b=list(map(lambda x:[]+[x],b))
     
        out=[[0]*len(b[0])]*len(a) 
        for i in range( len(out)):
            for j in range( len(out[0])):
                for k in range(len(a[0])):
                
                    out[i][j]+=(a[i][k]*b[k][j])
        return out
    def multiply(self,a,b):
        
        out=[]
        
        if type(b) == list:
            if type(a[0]) !=  list:
                a=list(map(lambda x:[]+[x],a))
            if type(b[0]) !=  list:
                b=list(map(lambda x:[]+[x],b))
            out=a
            for i in range( len(a)):
                for j in range( len(a[0])):
                    out[i][j]*=b[i][j]
        else:
           
            if type(a[0]) != list:
                a=list(map(lambda x:[]+[x],a))
            out=a
          
            for i in range( len(a)):
            
                for j in range( len(a[0])):
                    
                    out[i][j]*=b
            
       
        return out
    def add(self,a,n):
       
      
        out=[]
        if type(n)is list:
            if type(n[0]) is not list:
                n=list(map(lambda x:[]+[x],n))

            if type(a[0]) !=  list:
                a=list(map(lambda x:[]+[x],a))
            out=a
            for i in range(len(a)):
                
                for j in range(len(a[0])):
                 
                    out[i][j] += n[i][j];
                
        
        return out
    
  
    def cost(self,input,target):
       
        output=self.feed_foward(input)
        self.clear_nn()
        return sum(map(lambda x:(x[0]-x[1])**2,zip(output,target)))
        
    def train(self,input,target):
        
        output=self.feed_foward(input)
        
        errors=list(map(lambda x:x[0]-x[1],zip(target,output)))
   
     
        gradient=list(map(dsigmoid,output))
        gradient=self.multiply(gradient,errors)
        gradient=self.multiply(gradient,self.learning_rate)
        
        l=len(self.nn)-2
        while l>=0:
        
            
            layer_t=self.transpose(self.nn[l])
            deltGrad=self.multiplyOne(gradient,layer_t)
            

            self.weights[l]=self.add(self.weights[l],self.transpose(deltGrad))
            self.bias[l+1]=flatten_list(self.add(self.bias[l+1],gradient))
            if l==0:break   
            weight_t=self.transpose(self.weights[l])  
            errors=flatten_list(self.multiplyOne(errors,weight_t))
            
            gradient=list(map(dsigmoid,self.nn[l]))
           
            gradient=self.multiply(gradient,errors)
           
            gradient=self.multiply(gradient,self.learning_rate)
            l-=1
        self.clear_nn()

      
