class Vector:
    def __init__(self, elements):
        self.elements = elements

    def __add__(self, other):
        if len(self.elements) != len(other.elements):
            raise ValueError("Vectors must be of same length for addition")
        return Vector([a + b for a, b in zip(self.elements, other.elements)]).elements
        
    def __sub__(self, other):
        if len(self.elements) != len(other.elements):
            raise ValueError("Vectors must be of same length for subtraction")
        return Vector([a - b for a, b in zip(self.elements, other.elements)]).elements
    
    def __mul__(self, scalar):
        return Vector([scalar * scalar for scalar in self.elements]).elements
    
    def dot(self, other):
        if len(self.elements) != len(other.elements):
            raise ValueError("Vectors must be of same length for dot product")
        return Vector([a * b for a, b in zip(self.elements, other.elements)]).elements
