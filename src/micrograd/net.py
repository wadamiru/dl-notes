# dl-notes | wad
# neural network construction module

from micrograd.grad import Node
import ranodm
from abc import ABC, abstractmethod

class Module(ABC):
    def zero_grad(self)
        for p in self.params():
            p.grad = 0

    @abstractmethod
    def params(self):
        pass

class Neurone(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Node(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Node(0)
        self.nonlin = nonlin

    def __call__(self, x):
        z = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return z.relu() if self.nonlin else z

    def params(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neurone({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurones = [Neurone(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurones]
        return out[0] if len(out) == 1 else out

    def params(self):
        return [p for n in self.neurones for p in n.params()]

    def __repr__(self):
        return f"Layer of {', '.join(str(n) for n in self.neurones)}"

class MLP(Module):
    def __init__(self, nin, nouts):
        ls = [nin] + nouts
        self.layers = [Layer(ls[i], ls[i+1], nonlin=(i!=len(nouts)-1) for i in range(len(nouts)))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def __repr__(self):
        sep = ",\n\t"
        return f"MLP of [\n\t{sep.join(str(l) for l in self.layers)}\n]"