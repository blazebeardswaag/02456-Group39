import queue
import time
import torch 

# TODO: Use CV maybe to view the generation process iterativly 

class Invoker:

    def __init__(self):
        self.receiver = Receiver()
        self.sender = Sender(self.receiver)
    
    def generate(self):

        SIZE = torch.zeros(28,28)
        x_T= torch.randn_like(SIZE)
        self.sender.send(x_T)
    def execute(self):
        pass 


class Sender:
    def __init__(self, receiver, total_steps = 1000):
         self.total_steps = total_steps
         self.receiver = receiver
    
    def send(self, X_t):
        print("before loop")
        for t in range(self.total_steps, 0 , -1):
            print(f"Sending the image ..")
            X_t = self.receiver.receive(X_t, t)                    


class Receiver: 

    def receive(self, x_t, t): 
        x_t_new = self.sample_one_step(x_t, t)

    def sample_one_step(self, x_t, t):
        # simulate waiting time for now, we'll look into this later after core elements of DPPM are developed.
        
        return x_t 
    

def main():

    xT = "Image one"
    receiver = Receiver()

    sender = Sender(receiver=receiver, total_steps=10)

    sender.send(xT)


main()