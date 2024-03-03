class Tensor:
    # Constructor
    def __init__(self, data):
        self.data = data # Elements of the tensor
        self.shape = self.get_shape() # Shape of the tensor

    # Function to calculate the shape of the tensor
    def get_shape(self):
        shape = []
        temp = self.data # Creating a temporary copy of the tensor so that we can manipulate it to get the shape
        while isinstance(temp, list):
            shape.append(len(temp))
            temp = temp[0]
        del temp

        return tuple(shape)
    
    # Converting the tensor into an iterable so that we can iterate over the elements
    def __iter__(self):
        if isinstance(self.data, list):
            return iter(self.data)
        else:
            return iter([self.data])
    
    # Overriding the + operator
    def __add__(self, other):
        # Element wise addition
        if isinstance(other, (int, float)):
            return Tensor([[element + other for element in row] for row in self.data]).data
        # Tensor addition
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(f"Tensor of shape {self.shape} cannot be added with Tensor of shape {other.shape}")
            return Tensor([[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(self.data, other.data)]).data
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for operation with type Tensor)")
    
    # Overriding the - operator
    def __sub__(self, other):
        # Element wise subtraction
        if isinstance(other, (int, float)):
            return Tensor([[element - other for element in row] for row in self.data]).data
        # Tensor subtraction
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(f"Tensor of shape {self.shape} cannot be subtracted with Tensor of shape {other.shape}")
            return Tensor([[x - y for x, y in zip(row1, row2)] for row1, row2 in zip(self.data, other.data)]).data
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for operation with type Tensor)")

    # Overriding the * operator    
    def __mul__(self, scalar):
        # ELement wise multiplication
        if isinstance(scalar, (int, float)):
            return Tensor([[element * scalar for element in row] for row in self.data]).data
        else:
            raise TypeError(f"Unsupported operand type : {type(scalar)} for operation with type : Tensor)")
    
    # Dot product
    def dot(self, other):
        if isinstance(other, list):
            if len(other) != self.shape[1]:
                raise ValueError(f"Length of the list {len(other)} does not match the number of columns in the tensor.")
            return Tensor([[sum(row[i] * other[i] for i in range(len(other))) for row in self.data]]).data
        else:
            raise TypeError("Unsupported operand types.")
    
    # Overriding the @ operator used for matrix multiplication
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type {type(other)}")
        elif self.shape[1] != other.shape[0]:
            raise ValueError(f"Unsupported shape {other.shape}")
        else:
            return Tensor([[sum(a * b for a,b in zip(row, col)) for col in zip(*other.data)] for row in self.data]).data
        
    def mean(self, axis = None):
        if axis is None:
            return sum([sum(row) for row in self.data]) / self.shape[0] * self.shape[1]
        elif isinstance(axis, int):
            if axis >= len(self.shape):
                raise ValueError(f"Axis {axis} is out of range")
            # Mean along the columns
            elif axis == 0:
                return [sum(col) / self.shape[0] for col in zip(*self.data)]
            elif axis == 1:
                return [sum(row) / self.shape[1] for row in self.data]