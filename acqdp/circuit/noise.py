from acqdp.circuit.circuit import Channel, Circuit
import numpy as np


def Depolarization(px=0.25, py=None, pz=None):
    """Single-qubit depolarizing channel.

    :param px: One of the three parameters to describe the single-qubit depolarizing channel.
    :type px: :class:`float`.
    :param py: One of the three parameters to describe the single-qubit depolarizing channel.
    :type py: :class:`float`.
    :param pz: One of the three parameters to describe the single-qubit depolarizing channel.
    :type pz: :class:`float`.
    :returns: :class:`Mixture` -- the single-qubit depolarizing channel.
    """
    if py is None:
        py = px
    if pz is None:
        pz = px
    res = Channel(
        1,
        np.array([[[[1 - px - py, 0], [0, 1 - 2 * pz - px - py]],
                   [[0, px + py], [px - py, 0]]],
                  [[[0, px - py], [px + py, 0]],
                   [[1 - px - py - 2 * pz, 0], [0, 1 - px - py]]]]))
    return res


def Dephasing(pz=0.5):
    """Single-qubit dephasing channel, also called the phase-damping channel.

    :param pz: The parameter to describe the single-qubit dephasing channel.
    :type pz: :class:`float`.
    :returns: :class:`Mixture` -- the single-qubit dephasing channel with given parameter.
    """
    return Depolarization(0, 0, pz)


def AmplitudeDampling(p=0.1):
    res = Channel(
        1,
        np.array([[[[1, 0], [0, np.sqrt(1 - p)]],
                   [[0, p], [0, 0]]],
                  [[[0, 0], [0, 0]],
                   [[np.sqrt(1 - p), 0], [0, 1 - p]]]]))
    return res


def add_noise(circuit, noise_channel):
    c = Circuit()
    for k in circuit.operations_by_name:
        operation = circuit.operations_by_name[k]['operation']
        qubits = circuit.operations_by_name[k]['qubits']
        time_step = circuit.operations_by_name[k]['time_step']
        new_op = Circuit() | operation
        for a in new_op._output_indices[0]:
            new_op.append(noise_channel, [a])
        c.append(new_op, qubits=qubits, time_step=time_step, name=k)
    return c
