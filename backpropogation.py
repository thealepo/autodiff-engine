import math

class Tensor:
    def __init__(self , data , _children=() , _op='' , label=''):
        self.data = data
        self.grad = 0.0
        self._op = _op
        self._backward = lambda: None # override this
        self._prev = set(_children)
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data})"
    
    def __add__(self , other):
        other = other if isinstance(other , Tensor) else Tensor(other)
        out = Tensor(self.data + other.data , (self,other) , '+')

        def _backward():
            # dy/dx = 1, we multiply by out.grad for respect to the chain rule
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward # overriding the function!

        return out
    
    def __mul__(self , other):
        other = other if isinstance(other , Tensor) else Tensor(other)
        out = Tensor(self.data * other.data , (self,other) , '*')

        def _backward():
            # dy/dx = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self , other):
        assert isinstance(other , (int,float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other , (self,) , f'**{other}')

        def _backward():
            # dy/dx = n*x^(n-1)
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self , other):
        return self + (-other)
    
    def __radd__(self , other):
        return self + other
    
    def __rmul__(self , other):
        return self * other
    
    def __truediv__(self , other):
        return self * other**-1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Tensor(t , (self,) , 'tanh')

        def _backward():
            # dy/dx = 1-tanh^2(x)
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        out = Tensor(math.exp(x) , (self,) , 'exp')

        def _backward():
            # dy/dx = e^x
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def sine(self):
        x = self.data
        out = Tensor(math.sin(x) , (self,) , 'sin')

        def _backward():
            # dy/dx = cos(x)
            self.grad += self.cosine().data * out.grad
        out._backward = _backward

        return out
    
    def cosine(self):
        x = self.data
        out = Tensor(math.cos(x) , (self,) , 'cos')

        def _backward():
            # dy/dx = -sin(x)
            self.grad += -self.sine().data * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        out = (-(self)).exp()
        out = Tensor(1 / (1 + out.data) , (self,) , 'sigmoid')

        def _backward():
            # dy/dx = sigmoid(1-sigmoid)
            self.grad = (out.data * (1-out.data)) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # dx/dx (final node)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()