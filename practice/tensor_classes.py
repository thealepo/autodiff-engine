import math

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
    
class Tensor_4:
    def __init__(self , data , _op='' , label=''):
        self.data = data
        self.grad = 0.0
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Tensor(data={self.data})"
    
    def __add__(self , other):
        other = other if isinstance(other , Tensor_4) else Tensor_4(other)
        out = Tensor_4(self.data + other.data , '+')
        return out
    
    def __mul__(self , other):
        other = other if isinstance(other , Tensor_4) else Tensor_4(other)
        out = Tensor_4(self.data * other.data , '*')
        return out
    
    def __pow__(self , other):
        assert isinstance(other , (int,float)), "only supporting int/float powers for now"
        out = Tensor_4(self.data**other , f'**{other}')
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self , other):
        return self + (-other)
    
    def __radd__(self , other): # other + self
        return self + other
    
    def __rmul__(self , other): # other * self
        return self * other
    
    def __truediv__(self , other): # self / other
        return self * other**-1
    
    # These don't have dunder methods
    def tanh(self):
        x = self.data 
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Tensor_4(t , 'tanh')
        return out
    
    def exp(self): # e^x
        x = self.data
        out = Tensor_4(math.exp(x) , 'exp')
        return out
    
    def sine(self):
        x = self.data
        out = Tensor_4(math.sin(x) , 'sin')
        return out
    
    def cosine(self):
        x = self.data
        out = Tensor_4(math.cos(x) , 'cos')
        return out
    
    def sigmoid(self):
        out = (-(self)).exp()
        out = Tensor_4(1 / (1 + out.data) , 'sigmoid')
        return out
    
a = Tensor_4(2.0, label='a')
b = Tensor_4(3.0, label='b')
c = Tensor_4(4.0, label='c')

d = (a + b) * c
e = d.exp()
f = d.sigmoid()
g = (a - b).tanh()
h = (c / a).sine()

print("d =", d.data)
print("e =", e.data)
print("f =", f.data)
print("g =", g.data)
print("h =", h.data)
print("=============================================")