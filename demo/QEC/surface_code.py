from acqdp.tensor_network import TensorNetwork
import numpy as np
from acqdp.circuit import CNOTGate, CZGate, Circuit, HGate, Measurement, PlusState, State, Trace, ZeroMeas, ZeroState
from demo.QEC.noise_model import add_idle_noise, add_noisy_surface_code


params = {
    'T_1_inv': 1 / 30000.0,
    'T_phi_inv': 1 / 60000.0,
    'p_axis': 1e-4,
    'p_plane': 5e-4,
    'delta_phi': 0.01,
    'T_g_1Q': 20.0,
    'T_g_2Q': 40.0,
    'tau_m': 300.0,
    'tau_d': 300.0,
    'gamma': 0,
    'alpha0': 4,
    'kappa': 1 / 250,
    'chi': 1.3 * 1e-3}

Z = TensorNetwork([0, 0])
Z.add_node('PH', [0], np.ones(2))
PlaceHolder = Measurement(1, Z, name='PH_nz')
TraceState = State(1, TensorNetwork([0, 0], bond_dim=2))

qubit_group_name = {
    (0, 0): 'D1',
    (0, 2): 'D2',
    (2, 0): 'D3',
    (2, 2): 'D4',
    (1, 1): 'X1',
    (1, 3): 'Z1',
    (3, 1): 'Z2',
    (3, 3): 'X2',
    (-1, -1): 'dummy'
}
qubit_groups = {group: [] for group in qubit_group_name.values()}
qubit_coords = ([(x * 2, y * 2) for x in range(3) for y in range(3)]
                + [(-1, 3)]
                + [(x * 2 + 1, y * 2 + 1) for x in range(2) for y in (range(3) if x % 2 else range(-1, 2))]
                + [(5, 1)])

for x, y in qubit_coords:
    qubit_groups[qubit_group_name[x % 4, y % 4]].append((x, y))


def add_CZ_gates(circuit, high_freq_group, low_freq_group):
    for q1 in high_freq_group:
        for q2 in low_freq_group:
            if abs(q1[0] - q2[0]) == 1 and abs(q1[1] - q2[1]) == 1:
                circuit.append(CZGate, [q1, q2])
                break


def x_stab_meas(circuit, measure_outcome=None, use_ndcompmeas=False):
    """This is currently only used to add a round of X-stabilier measurement circuit without noise."""

    for group in ['D1', 'D2', 'D3', 'D4', 'X1', 'X2']:
        for qubit in qubit_groups[group]:
            circuit.append(HGate, [qubit])

    # Time slot 1 ~ 4
    for flux_dance in [('D2', 'X1', 'X2', 'D3', 'D4'),
                       ('D1', 'X1', 'X2', 'D4', 'D3'),
                       ('D1', 'X2', 'X1', 'D4', 'D3'),
                       ('D2', 'X2', 'X1', 'D3', 'D4')]:

        for g1, g2 in [flux_dance[0:2], flux_dance[2:4]]:
            add_CZ_gates(circuit, qubit_groups[g1], qubit_groups[g2])

    # Time slot B
    for group in ['D1', 'D2', 'D3', 'D4', 'X1', 'X2']:
        for qubit in qubit_groups[group]:
            circuit.append(HGate, [qubit])

    # Time slot C
    if use_ndcompmeas:
        for group in ['X1', 'X2']:
            for qubit in qubit_groups[group]:
                circuit.append(PlaceHolder, [qubit])
                circuit.append(TraceState, [qubit])
    else:
        if measure_outcome is not None:
            for group in ['X1', 'X2']:
                for qubit in qubit_groups[group]:
                    circuit.append(measure_outcome[qubit], [qubit])
                    circuit.append(ZeroState, [qubit])
        else:
            for group in ['X1', 'X2']:
                for qubit in qubit_groups[group]:
                    circuit.append(ZeroMeas, [qubit])
                    circuit.append(ZeroState, [qubit])


def z_stab_meas(circuit, measure_outcome=None, use_ndcompmeas=False):
    """
    measure_outcome: a dictionary, from qubit_coords, to ZeroMeas/OneMeas

    This is currently only used to add a round of Z-stabilier measurement circuit without noise
    """

    for group in ['Z1', 'Z2']:
        for qubit in qubit_groups[group]:
            circuit.append(HGate, [qubit])

    # Time slot 5 ~ 8
    for flux_dance in [('D1', 'Z1', 'Z2', 'D4', 'D3'),
                       ('D2', 'Z2', 'Z1', 'D3', 'D4'),
                       ('D2', 'Z1', 'Z2', 'D3', 'D4'),
                       ('D1', 'Z2', 'Z1', 'D4', 'D3')]:
        for g1, g2 in [flux_dance[0:2], flux_dance[2:4]]:
            add_CZ_gates(circuit, qubit_groups[g1], qubit_groups[g2])

    for group in ['Z1', 'Z2']:
        for qubit in qubit_groups[group]:
            circuit.append(HGate, [qubit])

    if use_ndcompmeas:
        for group in ['Z1', 'Z2']:
            for qubit in qubit_groups[group]:
                circuit.append(PlaceHolder, [qubit])
                circuit.append(TraceState, [qubit])
    else:
        if measure_outcome is not None:
            for group in ['Z1', 'Z2']:
                for qubit in qubit_groups[group]:
                    circuit.append(measure_outcome[qubit], [qubit])
                    circuit.append(ZeroState, [qubit])
        else:
            for group in ['Z1', 'Z2']:
                for qubit in qubit_groups[group]:
                    circuit.append(ZeroMeas, [qubit])
                    circuit.append(ZeroState, [qubit])


def initial_state(coord=(-100, -100)):
    """prepare a maximally entangled state between surface code and ancilla qubit.

    First prepare |0>_surf |+>_anc, then do a few CNOT
    """
    c = Circuit()

    for q in qubit_coords:
        c.append(ZeroState, [q])

    x_stab_meas(c)

    c.append(PlusState, [coord])
    c.append(CNOTGate, [coord, (0, 0)])
    c.append(CNOTGate, [coord, (2, 0)])
    c.append(CNOTGate, [coord, (4, 0)])
    return c


def final_measurement(circuit: Circuit):
    for q in [(1, 1), (1, -1), (3, 3), (3, 1), (-1, 3), (5, 1), (1, 3), (3, 5)]:
        circuit.append(Trace, [q])
        circuit.append(ZeroState, [q])
    x_stab_meas(circuit, use_ndcompmeas=True)  # Add final noiseless X-stabilizer measurements
    for qubits in [[(1, 1), (0, 0)], [(3, 3), (4, 4)], [(-1, 3), (0, 4)], [(5, 1), (4, 0)]]:
        circuit.append(CZGate, qubits)
        circuit.append(Trace, [qubits[0]])
        circuit.append(TraceState, [qubits[0]])

    z_stab_meas(circuit, use_ndcompmeas=True)  # Add final noiseless Z-stabilizer measurements
    for qubits in [[(1, -1), (0, 0)], [(3, 1), (4, 0)], [(1, 3), (0, 4)], [(3, 5), (4, 4)]]:
        circuit.append(CNOTGate, qubits)
        circuit.append(Trace, [qubits[0]])
        circuit.append(TraceState, [qubits[0]])
    return circuit


def surface_code_tensor_network(num_layers=2, params=params):
    noisy_meas_circ = Circuit()
    end_time = 0
    for _ in range(num_layers):
        end_time = add_noisy_surface_code(noisy_meas_circ, qubit_coords, time=end_time, params=params)
    add_idle_noise(noisy_meas_circ, params=params)
    d = final_measurement(noisy_meas_circ)
    init_state = initial_state()
    c_prob = init_state | d | initial_state(coord=(-101, -101)).adjoint()
    tn = c_prob.tensor_density.expand(recursive=True)
    for node_name in tn.nodes_by_name:
        if node_name[-1] == 'PH':
            tn.remove_node(node_name)
    return tn


def surface_code_tensor_network_with_syndrome(syndrome=None, num_layers=2, params=params):
    if syndrome is None:
        syndrome = [0] * (8 * (num_layers + 1))
    noisy_meas_circ = Circuit()
    e_ro = params.get('e_ro', 0.01)
    butterfly = np.array([[1 - e_ro, e_ro], [e_ro, 1 - e_ro]])
    end_time = 0
    for _ in range(num_layers):
        end_time = add_noisy_surface_code(noisy_meas_circ,
                                          qubit_coords,
                                          time=end_time,
                                          params=params)
    add_idle_noise(noisy_meas_circ, params=params)
    d = final_measurement(noisy_meas_circ)
    init_state = initial_state()
    c_prob = init_state | d | initial_state(coord=(-101, -101)).adjoint()
    tn = c_prob.tensor_density.expand(recursive=True)
    cnt = 0
    for node_name in tn.nodes_by_name:
        if node_name[0] == 'PH' or node_name[0] == 'PH_nz':
            if syndrome is not None:
                if node_name[0] == 'PH':
                    tn.update_node(node_name, butterfly[syndrome[cnt]])
                else:
                    tn.fix_edge(tn.network.nodes[(0, node_name)]['edges'][0][0],
                                syndrome[cnt])
            cnt += 1
    return tn
