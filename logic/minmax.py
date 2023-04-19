import torch
import torch.nn as nn
import torch.nn.functional as F

# network definition
class TwoLayers(nn.Module):
    def __init__(self):
        super(TwoLayers, self).__init__()

        self.fc1 = nn.Linear(3,3)
        self.fc2 = nn.Linear(3,3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sign(x)
        x = self.fc2(x)
        x = torch.sign(x)
        return x

    def set_weight_bias( self, w1, b1, w2, b2 ):
        self.fc1.weight.data = w1.transpose(0,1).clone()
        self.fc1.bias.data = b1.clone()
        self.fc2.weight.data = w2.transpose(0,1).clone()
        self.fc2.bias.data = b2.clone()

N=4
X = torch.rand( N, 3 )
net = TwoLayers()

crr = 0

### find MIN ###
print("find MIN:")
w1 = torch.tensor(
 [[+1,0,0],
  [0,+1,0],
  [0,0,+1]], dtype=torch.float32 )
b1 = torch.tensor(
 [0,0,0], dtype=torch.float32 )

w2 = torch.tensor(
 [[1,0,0],
  [0,1,0],
  [0,0,1]], dtype=torch.float32 )
b2 = torch.tensor(
 [0,0,0], dtype=torch.float32 )

net.set_weight_bias( w1, b1, w2, b2 )
Y = net(X)
Y = Y.to(dtype=torch.int32)
for i in range(N):
    t = -torch.ones( 3, dtype=torch.int32 )
    ind = X[i,:].argmin()
    t[ind] = +1
    print( "{:.3f} {:.3f} {:.3f}".format(X[i,0],X[i,1],X[i,2]), end=" -> ")
    print( "{:+d} {:+d} {:+d}".format(Y[i,0],Y[i,1],Y[i,2]), end=" ")
    print( "[{:+d} {:+d} {:+d}]".format(t[0], t[1], t[2] ), end=" " )
    if( (Y[i,:] == t).all() ): print("OK"); crr += 1
    else: print("NG")
print()

### find MAX ###
print("find MAX:")
w1 = torch.tensor(
 [[1,0,0],
  [0,1,0],
  [0,0,1]], dtype=torch.float32 )
b1 = torch.tensor(
 [0,0,0], dtype=torch.float32 )

w2 = torch.tensor(
 [[1,0,0],
  [0,1,0],
  [0,0,1]], dtype=torch.float32 )
b2 = torch.tensor(
 [0,0,0], dtype=torch.float32 )

net.set_weight_bias( w1, b1, w2, b2 )
Y = net(X)
Y = Y.to(dtype=torch.int32)
for i in range(N):
    t = -torch.ones( 3, dtype=torch.int32 )
    ind = X[i,:].argmax()
    t[ind] = +1
    print( "{:.3f} {:.3f} {:.3f}".format(X[i,0],X[i,1],X[i,2]), end=" -> ")
    print( "{:+d} {:+d} {:+d}".format(Y[i,0],Y[i,1],Y[i,2]), end=" ")
    print( "[{:+d} {:+d} {:+d}]".format(t[0], t[1], t[2] ), end=" " )
    if( (Y[i,:] == t).all() ): print("OK"); crr += 1
    else: print("NG")
print()

if( crr >= 8 ): print( "PASSED" )
else: print( "FAILED" )
