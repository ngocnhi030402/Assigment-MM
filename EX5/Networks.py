
import numpy as  np

class Network:
    def __init__(self):
        self.layers=[]
        self.loss=None
        self.loss_prime=None
    
    def add(self,layer):
        self.layers.append(layer)
        
    
    def setup_loss(self,loss,loss_prime):
        self.loss=loss
        self.loss_prime=loss_prime
    
    
    def fit(self,X_train,Y_train,is_R=True,learning_rate=0.0001,epochs=100):
        n=len(X_train)
        for i in range(epochs):
            err=0.0
            for j in range(n):
                output=X_train[j]
                # forward
                for layer in self.layers:
                    output=layer.forward_propagation(output)

                temp=np.array(X_train[j]).T
            

                if(is_R): # R 
                    temp=temp[0]
                    temp=temp.reshape((-1,1))
                    #estimate error of each sample
                    err += self.loss(Y_train[j],(output*0.001)+temp)
                    #calculate error to backward
                    error=self.loss_prime(Y_train[j],(output*0.001)+temp)
                else:
                    temp=temp[1]
                    temp=temp.reshape((-1,1))
                    #estimate error of each sample
                    err += self.loss(Y_train[j],(output*0.001)+temp)
                    #calculate error to backward
                    error=self.loss_prime(Y_train[j],(output*0.001)+temp)
                                        
                for layer in reversed(self.layers):
                    error= layer.backward_propagation(error,learning_rate)
            
            
            print('epochs : %d/%d err=%f'%(i+1,epochs,err))  

        