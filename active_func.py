from abc import abstractmethod


class ActiveFunc:
    @abstractmethod
    def calculate(self, x):
        pass
    
    @abstractmethod
    def calculate_derivative(self, x):
        pass
