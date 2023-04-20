import torch
import torch.nn as nn
import torch.nn.functional as F

# network definition
class SingleLayer(nn.Module):
    def __init__(self):
        super(SingleLayer, self).__init__()

        self.fc = nn.Linear(2,1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sign(x)
        return x

    def set_weight_bias( self, w, b ):
        self.fc.weight.data = torch.tensor( [w], dtype=torch.float32 )
        self.fc.bias.data = torch.tensor( b, dtype=torch.float32 )

net = SingleLayer()
X = torch.tensor( [ [-1,-1], [-1,+1], [+1,-1], [+1,+1] ], dtype=torch.float32 )


crr = 0

### logical AND ###
# edit those three parameters
weight = [0.5, 0.5]
bias = [0.0]
# edit those three parameters

print( "Logical AND" )
T = [-1,-1,-1,+1]
net.set_weight_bias( weight, bias )
Y = net(X)
for x,y,t in zip(X.detach().numpy(),Y.detach().numpy(),T):
    if( y == t ): r = 'OK'; crr += 1;
    else: r = 'NG'
    print( "{:+d} {:+d} -> {:+d} {}".format( int(x[0]), int(x[1]), int(y[0]), r ) )
print()


### logical OR ###
# edit those three parameters
weight = [0.5, 0.5]
bias = [0.0]
# edit those three parameters

print( "Logical OR" )
T = [-1,+1,+1,+1]
net.set_weight_bias( weight,bias )
Y = net(X)
for x,y,t in zip(X.detach().numpy(),Y.detach().numpy(),T):
    if( y == t ): r = 'OK'; crr += 1;
    else: r = 'NG'
    print( "{:+d} {:+d} -> {:+d} {}".format( int(x[0]), int(x[1]), int(y[0]), r ) )
print()



### logical NOT x1 ###
# edit those three parameters
weight = [0.5, 0.5]
bias = [0.0]
# edit those three parameters

print( "Logical NOT x1" )
T = [+1,+1,-1,-1]
net.set_weight_bias( weight, bias )
Y = net(X)
for x,y,t in zip(X.detach().numpy(),Y.detach().numpy(),T):
    if( y == t ): r = 'OK'; crr += 1;
    else: r = 'NG'
    print( "{:+d} {:+d} -> {:+d} {}".format( int(x[0]), int(x[1]), int(y[0]), r ) )
print()

if( crr >= 12 ): print( 'PASSED' )
else: print( 'FAILED' )

