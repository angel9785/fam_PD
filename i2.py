from multiprocessing import Process, Queue
import time
import sys

def two(q):
    for i in range(0,1000):
        while not q.empty():
            m=q.get()
            print (m)
q = Queue()
g = Process(target=two, args=(q,))
g.start() 

for i in range(0,10):
    q.put(i)
    
g.join()
