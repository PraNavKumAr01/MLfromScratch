import sys
sys.path.append('../')

class Tensor:
    # Constructor
    def __init__(self, data):
        self.data = data # Elements of the tensor
        self.shape = self.get_shape() # Shape of the tensor

    # Function to return len(tensor)
    def __len__(self):
        return self.shape[0]
    
    # Calculating the dimension of the tensor
    def n_dim(self):
        return len(self.shape)
    
    # Flattening a multi dimensional tensor into a 1D tensor
    def flatten(self):
        return [item for row in self.data for item in row]
    
    # Calculating the max of a Tensor according to the axis
    def max(self, axis = None):
        # Max of the entire tensor
        if axis is None:
            return max(self.flatten())
        # Max of every column
        if axis == 0:
            return Tensor([max(col) for col in zip(*self.data)]).data
        # Max of every row
        if axis == 1:
            return Tensor([max(row) for row in self.data]).data

    # Calculating the min of a Tensor according to the axis 
    def min(self, axis = None):
        # Min of the entire tensor
        if axis is None:
            return min(self.flatten())
        # Min of every column
        if axis == 0:
            return Tensor([min(col) for col in zip(*self.data)]).data
        # Min of every row
        if axis == 1:
            return Tensor([min(row) for row in self.data]).data

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
        if isinstance(other, Tensor):
            if other.n_dim() != self.n_dim():
                raise ValueError(f"Dimension mismatch")
            elif other.n_dim() == 1:
                if len(self.data) != len(other.data):
                    raise ValueError(f"Length of tensors must be equal")
                else:
                    return sum(a * b for a, b in zip(self.data, other.data))
            elif other.n_dim() == 2:
                return Tensor([[sum(a * b for a,b in zip(row, col)) for col in zip(*other.data)] for row in self.data]).data
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
    
    # Calculating the mean of the Tensor
    def mean(self, axis = None):
        if self.n_dim() == 1:
            return sum(self.data) / len(self.data)
        else:
            if axis is None:
                total_sum = sum(sum(row) for row in self.data)
                return total_sum / (self.shape[0] * self.shape[1])
            elif isinstance(axis, int):
                if axis >= len(self.shape):
                    raise ValueError(f"Axis {axis} is out of range")
                # Mean along the columns
                elif axis == 0:
                    return [sum(col) / self.shape[0] for col in zip(*self.data)]
                elif axis == 1:
                    return [sum(row) / self.shape[1] for row in self.data]
                
    # Transposing a multi-dimensional Tensor
    def transpose(self):
        if self.n_dim() == 1:
            return self.data
        else:
            return Tensor([list(row) for row in zip(*self.data)]).data