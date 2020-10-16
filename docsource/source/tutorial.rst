Tutorial
===========

In the tutorial, we will show you two examples of using `ACQDP`. The first one is to use the tensor network functionalities to experiment with tensor network states. The second one is to use the circuit library to experiment with the fidelity of GHZ states under various noise models.

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

MPS State
---------
In this section, we create two random ring-MPS states and calculate their fidelity.

An MPS state is a quantum state formulated as a centipede-shaped tensor network. We first define a random MPS state on a ring, with bond dimension and number of qubits given as the input:

.. code-block:: python

    from acqdp.tensor_network import TensorNetwork, normalize
    import numpy

    def MPS(num_qubits, bond_dim):
        a = TensorNetwork()
        for i in range(num_qubits):
            tensor = numpy.random.normal(size=(2, bond_dim, bond_dim)) +\
                        1j * numpy.random.normal(size=(2, bond_dim, bond_dim))
            a.add_node(i, edges=[(i, 'o'), (i, 'i'), ((i+1) % num_qubits, 'i')], tensor=tensor)
            a.open_edge((i, 'o'))
        return normalize(a)

This constructs an MPS state of the following form, where the internal connections `(i, 'i')` have bond dimension `bond_dim` and the outgoing wires `(i, 'o')` have bond dimension 2 representing a qubit system.

.. image:: MPS.pdf
  :width: 700
  :alt: An illustration of a ring-MPS state

Note that normalize() computes the frobenius norm of the tensor network, which already involves tensor network contraction.

For a further break down, in the code we first defined a tensor network:

.. code-block:: python

    a = TensorNetwork()

Each tensor `i` is of shape `(2, bond_dim, bond_dim)`. We add the tensor into the tensor network by specifying its connection in the tensor network:

.. code-block:: python

    a.add_node(i, edges=[(i, 'o'), (i, 'i'), ((i+1) % num_qubits, 'i')], tensor=tensor)

Finally, the outgoing edges `[(i, 'o')]` needs to be opened:

.. code-block:: python

    a.open_edge((i, 'o'))

This allows us to get two random MPS states:

.. code-block:: python

    a = MPS(10, 3)
    b = MPS(10, 3)

To calculate the fidelity of the two states, we put them into a single tensor network representing their inner product:

.. code-block:: python

    c = TensorNetwork()
    c.add_node('a', range(10), a)
    c.add_node('b', range(10), ~b)

Here, The tensor network `c` is constructed by adding the two tensor valued objects `a` and `~b` into the tensor network. The outgoing edges of `a` are identified as 0 to 9 in `c`, and that matches the outgoing edges of `b`. As no open edges is indicated in `c`, it sums over all the indices 0 to 9 and yield the inner product of `a` and `b`. (Note that the complex conjugate of b is added instead of b itself.)

This tensor network `c` takes two tensor valued objects `a` and `b` which are not necessarily tensors. This is a feature of the ACQDP: components in tensor networks do not have to be tensors, which allows nested structures of tensor networks to be easily constructed. The fidelity is then the absolute value of the inner product:

.. code-block:: python

    print("Fidelity = {}".format(numpy.abs(c.contract()) ** 2))


GHZ State
---------
The next example features our circuit module, which allows simulation of quantum computation supported by the powerful tensor network engine. A priliminary noise model is also included.

A :math:`n`-qubit GHZ state, also known as “Schroedinger cat states” or just “cat states”, are defined as :math:`\frac{1}{\sqrt2}\left(|0\rangle^{\otimes n}+|1\rangle^{\otimes n}\right)`. A :math:`n`-qubit GHZ state can be prepared by setting the first qubit :math:`|+\rangle`, and apply CNOT gate sequentially from the first qubit to all the other qubits. In ACQDP, we first define the circuit preparing the GHZ state:

.. code-block:: python

    from acqdp.circuit import Circuit, HGate, CNOTGate, ZeroState

    def GHZState(n):
        a = Circuit().append(ZeroState, [0]).append(HGate, [0])
        for i in range(n - 1):
            a.append(ZeroState, [i + 1])
            a.append(CNOTGate, [0, i + 1])
        return a

A GHZ state then can be constructed upon calling :math:`GHZState(n)`. A 4-qubit GHZ state is then

.. code-block:: python

    a = GHZState(4)

`a` is right now a syntactic representation of the GHZ state as a gate sequence. To examine the state as a tensor representing the pure state vector,

.. code-block:: python

    a_tensor = a.tensor_pure
    print(a_tensor.contract())

gives the output

.. code-block:: python

    array([[[[0.70710678, 0.        ],
            [0.        , 0.        ]],

            [[0.        , 0.        ],
            [0.        , 0.        ]]],


          [[[0.        , 0.        ],
            [0.        , 0.        ]],

            [[0.        , 0.        ],
            [0.        , 0.70710678]]]]).

The `tensor_pure` of a circuit object returns the tensor network representing it as a pure operation, i.e. a state vector, an isometry, or a projective measurement. In this case we do get the state vector; the density matrix will be returned if we choose to contract the `tensor_density`.

We are now interested in how the fidelity is preserved under simplified noise models.

.. code-block:: python

    from acqdp.circuit import add_noise, Depolarization
    b = add_noise(a, Depolarization(0.01))

The quantum state `b` representing noisy preparation of the GHZ state is no longer pure. To compute the fidelity of `b` and `a`, we compute the probability of postselecting `b` on the state `a`, i.e. concatenate `b` with `~a`:

.. code-block:: python

    c = (b | ~a)
    print(c.tensor_density.contract())

which gives the result `0.7572548016539656`.

The landscape of the fidelity with respect to the depolarization strength is given in the following figure:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 0.25, 0.01)
    y10 = [1.0000000000000002, 0.47542461105055484, 0.23487071438231183, 0.12115735903392826, 0.0652719534015429, 0.036617680257557426, 0.021320371501238007, 0.012865239407674912, 0.008058243176286981, 0.005262103481307746, 0.0036037915311682234, 0.002603081578208944, 0.0019895879912888935, 0.0016082666392452137, 0.001368731067752561, 0.0012173606455788866, 0.001121708636479879, 0.001061702644494669, 0.0010246415790509495, 0.0010023252937947874, 0.0009893837155440777, 0.0009822806445391288, 0.0009786976371788851, 0.0009771335153513456, 0.0009766284332079827]
    y5 = [1.0000000000000002, 0.6999844145588493, 0.4942690316849154, 0.35324185097666616, 0.25638313196195456, 0.1895935117363198, 0.14325927357443888, 0.11086721247093151, 0.08802473314645329, 0.07177520425921743, 0.060125720084480076, 0.05172560640086923, 0.04565037199901648, 0.041258297754298824, 0.03809626462551216, 0.035838408954879976, 0.03424630001392201, 0.033143002111564755, 0.032395966801493, 0.03190548171704453, 0.031596601835520065, 0.03141327605432126, 0.03131388459310738, 0.03126771494539828, 0.0312520931895467]
    y9 = [1.0000000000000002, 0.5132381922266505, 0.27166900372328906, 0.14908909496638956, 0.08499787250437944, 0.0503080570674666, 0.030859873469816123, 0.01960047090479715, 0.01290249474675171, 0.008829579127738789, 0.0063089453125740646, 0.0047265524768867915, 0.003721592330004327, 0.0030776535845843216, 0.0026627047793805945, 0.002394907394164172, 0.0022227022250304526, 0.0021130478204492353, 0.0020444111128750074, 0.002002555102440312, 0.0019779765982536465, 0.0019643161156803883, 0.0019573399295379776, 0.0019542600701634096, 0.0019532566580686113]
    y8 = [1.0000000000000002, 0.5542898030047229, 0.31472871323487583, 0.18401831669889898, 0.11114615113427825, 0.06942896286520428, 0.04485003004992563, 0.029956265215634527, 0.020702906622970918, 0.014834003362939607, 0.011051355010576764, 0.008584138930327863, 0.006961588246175839, 0.005889338848034959, 0.005179768537503058, 0.004711361749931511, 0.004404297210961834, 0.0042054590581549825, 0.004079094318600177, 0.004000921928759586, 0.003954371863209565, 0.003928141910971797, 0.003914569170475958, 0.003908506219259532, 0.003906512899442519]
    y7 = [1.0000000000000002, 0.5988759886636054, 0.36520885830779715, 0.22787751587652708, 0.14603245642264728, 0.09634967216232196, 0.06553695207133448, 0.04599360960731351, 0.0333314264793019, 0.024975471552979505, 0.019381152335915943, 0.015597642695686274, 0.013023633249363637, 0.011269075600120797, 0.010075326362177048, 0.009267762911382808, 0.00872681002216888, 0.008369704573638986, 0.008138729130284681, 0.007993461983708823, 0.007905579323118082, 0.00785530286847755, 0.007828916935169336, 0.007816984595525935, 0.007813024965493753]
    y6 = [1.0000000000000002, 0.6473219489080633, 0.42449925291790536, 0.2831859627887721, 0.19290889674361097, 0.13461252752812014, 0.09645080103190168, 0.07108101348153528, 0.05394736164157386, 0.04220677469037928, 0.03406511183233029, 0.028371421880545485, 0.02437196355627255, 0.021561679208785108, 0.019594356328221877, 0.018227799850946545, 0.017289803616474746, 0.016656516040107903, 0.01623827845499067, 0.015970060933913902, 0.01580480241205252, 0.015708638639341736, 0.015657390490663555, 0.01563391348132136, 0.015626048264106986]
    plt.xlabel('Noise level')
    plt.ylabel('Fidelity')
    plt.plot(x, y5, label='5-qubit GHZ Circuit')
    plt.plot(x, y6, label='6-qubit GHZ Circuit')
    plt.plot(x, y7, label='7-qubit GHZ Circuit')
    plt.plot(x, y8, label='8-qubit GHZ Circuit')
    plt.plot(x, y9, label='9-qubit GHZ Circuit')
    plt.plot(x, y10, label='10-qubit GHZ Circuit')
    plt.title("Demonstration of Noise Model Study on the ACQDP")
    plt.legend()
    plt.show()
