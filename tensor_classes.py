
class Tensor_1:
    def __init__(self , data):
        self.data = data
        self.grad = 0.0

class Tensor_2:
    
    def __init__(self , data , _op='' , label=''):
        self.data = data
        self.grad = 0.0
        self._op = _op #operation
        self.label = label
    def __repr__(self):
        return f"Tensor(data={self.data})"
    
tensor = Tensor_2(2.0 , label='x')
print(tensor.data)
print("=============================================")

class Tensor_3:
    def __init__(self , data , _op='' , label=''):
        self.data = data
        self.grad = 0.0
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Tensor(data={self.data})"
    
    def __add__(self , other):
        # i want to be able to call Tensor(4) + 5 = Tensor(9), for example
        other = other if isinstance(other , Tensor_3) else Tensor_3(other)
        out = Tensor_3(self.data + other.data , '+')
        return out
    def __mul__(self , other):
        # left multiplication
        other = other if isinstance(other , Tensor_3) else Tensor_3(other)
        out = Tensor_3(self.data * other.data , '*')
        return out
    def __neg__(self , other):
        # this would be y = -x, for example
        return self * -1
    def __sub__(self , other):
        return self + (-other)