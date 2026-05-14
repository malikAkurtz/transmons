from __future__ import annotations

import numpy as np

from Operator import Operator

class Expression():
    
    def __add__(self, other):
        return Add(self, other)
    
    def __mul__(self, other):
        return Mul(self, other)
    
    def __pow__(self, other):
        return Pow(self, other)
    
    @staticmethod
    def realize(expression: Expression, n_charge: int):        
        if isinstance(expression, Scalar):
            return expression.value
        elif isinstance(expression, Op) and expression.name == "n":
            # Symmetric integer charge ladder n in {-N, ..., +N}.
            N = int((n_charge - 1) / 2)
            
            charge_values = np.array([k for k in range(-N, N + 1)])
            return np.diag(charge_values)
        
        elif isinstance(expression, Cos) and isinstance(expression.a, Op) and (expression.a.name == "phi"):
            # Symmetric integer charge ladder n in {-N, ..., +N}.
            N = int((n_charge - 1) / 2)
            
            upper_lower_matrix = np.zeros((n_charge, n_charge))
            for i in range(-N, N):
                r = i + N
                upper_lower_matrix += np.outer(np.eye(n_charge)[r], np.eye(n_charge)[r+1]) + np.outer(np.eye(n_charge)[r+1], np.eye(n_charge)[r])
            return upper_lower_matrix / 2
        
        elif isinstance(expression, Add):
            a = Expression.realize(expression.a, n_charge)
            b = Expression.realize(expression.b, n_charge)
            return a + b
        
        elif isinstance(expression, Mul):
            a = Expression.realize(expression.a, n_charge)
            b = Expression.realize(expression.b, n_charge)
            return a * b
            
        elif isinstance(expression, Pow):
            a = Expression.realize(expression.a, n_charge)
            b = Expression.realize(expression.b, n_charge)
            if isinstance(a, Operator):
                result = np.eye(n_charge)
                for i in range(b):
                    result @=  a
                return result
            else:
                return a**b
            
        raise TypeError(f"Don't know how to realize: {expression}")

        
# Interal node
class Add(Expression):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def __str__(self):
        return f"{self.a} + {self.b}"

# Interal node
class Mul(Expression):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def __str__(self):
        return f"{self.a} * {self.b}"

# Interal node
class Pow(Expression):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def __str__(self):
        return f"{self.a}**{self.b}"
    
# Interal node
class Cos(Expression):
    def __init__(self, a):
        self.a = a

    def __str__(self):
        return f"cos({self.a})"
    
# Leaf node
class Scalar(Expression):
    def __init__(self, value: float):
        self.value = value
        
    def __str__(self):
        return f"{self.value}"
        
# Leaf node
class Op(Expression):
    def __init__(self, name: str):
        self.name = name
        
    def __str__(self):
        return self.name