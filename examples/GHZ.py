from acqdp.circuit import add_noise, Depolarization
from acqdp.circuit import Circuit, HGate, CNOTGate, ZeroState


def GHZState(n):
    a = Circuit().append(ZeroState, [0]).append(HGate, [0])
    for i in range(n - 1):
        a.append(ZeroState, [i + 1])
        a.append(CNOTGate, [0, i + 1])
    return a


a = GHZState(10)
b = add_noise(a, Depolarization(0.01))
print((b | a.adjoint()).tensor_density.contract())
