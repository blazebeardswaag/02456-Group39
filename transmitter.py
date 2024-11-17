import queue
import time

# TODO: Use CV maybe to view the generation process iterativly 

class Invoker:

    def __init__(self, receiver, sender):
        self.receiver = receiver
        self.sender = sender
    
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
            X_t = self.receiver.receive(X_t)                    




class Receiver: 

    def receive(self, x_t): 

        x_t_new = self.sample_one_step(x_t)

    def sample_one_step(self, x_t):
        # simulate waiting time for now, we'll look into this later after core elements of DPPM are developed.
        time.sleep(0.5)
        print(f"finished processing image")
        return x_t 
    

def main():

    xT = "Image one"
    receiver = Receiver()

    sender = Sender(receiver=receiver, total_steps=10)

    sender.send(xT)


main()