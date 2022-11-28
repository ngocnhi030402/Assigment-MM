import numpy as np
import matplotlib.pyplot as plt
import math

Romeo1 = []
Juliet1 = []
Romeo2 = []
Juliet2 = []
t = []
n = 0.3

def f(t0, R0, J0):
    return (-1)*R0  - 3*J0
def g(t0, R0, J0):
    return (-2)*R0 - 1*J0
def R(t):
    return math.exp(-3.44948974*t)*(2.837)  + (-0.837)*math.exp(1.44948974*t)
def J(t):
    return math.exp(-3.44948974*t)*(2.314992) + 0.68299199999*math.exp(1.44948974*t)



while n < 0.99:
    n += 0.01
    t.append(n)
    
def ExplicitEuler( t0, R0, J0):
   
    for i in t:
            R1 = R0 + f(t0, R0, J0) * i
            J1 = J0 + g(t0, R0, J0) * i
            t1 = t0 + i
            Romeo1.append(R1)
            Juliet1.append(J1)


    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Approximation of Romeo and Juliet')

    ax.plot(t, Romeo1, color = 'red', label ='Romeo');
    ax.plot(t, Juliet1, color = 'blue', label = 'Juliet');
    fig.legend(['Romeo', 'Juliet'], loc = 'upper right');

    plt.show()
    
                
ExplicitEuler(1,2,5)




def ImpicitEuler(t0,R0,J0):

   for i in t:
     a1 =1-(2*i)
     b2 =1-(2*i)
     a2=1
     b1=1
     c1=R0
     c2=J0
     t1=t0+i
     D = a1 * b2 - a2 * b1
     Dx = c1 * b2 - c2 * b1
     Dy = a1 * c2 - a2 * c1
     R1=Dx/D
     J1=Dy/D
     Romeo2.append(R1)
     Juliet2.append(J1)
     

   fig, ax = plt.subplots() 
     
   ax.set_xlabel('Time') 
   ax.set_ylabel('Approximation of Romeo and Juliet') 

   ax.plot(t, Romeo2, color = 'red', label='Romeo') 
   ax.plot(t, Juliet2, color = 'blue', label = 'Juliet') 
   fig.legend(['Romeo', 'Juliet'], loc='upper right')

   plt.show()
        

        
ImpicitEuler(1,2,5)