from multiprocessing import Process
import os
import time
from queue import Queue 



def f(name):
    for i in range(0,100):
        print('hello_girl', name)
        time.sleep(0.005)

def g(name):
    for i in range(0,100):
        print('hello_boy', name)
        time.sleep(0.005)
if __name__ == '__main__':     
    p = Process(target=f, args=('fereshteh',))
    q = Process(target=g, args=('milad',))
    p.start()
    q.start()   
    p.join()     
    q.join()



    

    



   
       

        
        

