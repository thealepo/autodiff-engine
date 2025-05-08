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