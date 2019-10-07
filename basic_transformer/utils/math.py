import numpy as np


class AvgContainer:
    """
    Average container
    """
    def __init__(self):
        self.__avg = 0
        self.__size = 0
    
    @property
    def avg(self):
        return self.__avg
    
    @property
    def size(self):
        return self.__size
    
    def update(self, values):
        incoming_size = len(values) 
        incoming_total = sum(values)
        new_size = self.__size + incoming_size
        incoming_avg = np.sum(values) / new_size
        old_updated_avg = self.__avg * (self.__size / float(new_size))
        # update
        self.__avg = old_updated_avg + incoming_avg
        self.__size = new_size
        
    def reset(self):
        self.__avg = 0
        self.__size = 0
        
        
def normalize_axis_1(x):
    x_normalized = x.T / x.sum(axis=1)
    x_normalized = x_normalized.T
    return x_normalized