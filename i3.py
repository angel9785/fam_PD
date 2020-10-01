from multiprocessing import Process, Queue
import time
import sys

def one(q,h):
    while True:
        
                    
        m1=q.get() 
        if(m1=='done'):
            break           
        h1=m1*2
        h.put(h1)
    h.put('done')

def two(h):
    while True:   
                 
        m2=h.get()
        if(m2=='done'):
            break                  
        print(m2)
q = Queue()
h = Queue()
g = Process(target=one, args=(q,h,))
g.start() 
p = Process(target=two, args=(h,))
p.start() 

for i in range(0,10):
    q.put(i)
    
q.put('done')
p.join()     
g.join()
