from acqdp.tensor_network import TensorNetwork
from acqdp.circuit import CZGate, Channel, Circuit, Depolarization, Diagonal, HGate, Measurement, State
import numpy as np

default_params = {
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

X_TALK_GAMMA = False


class IdleGate(Channel):
    def __init__(self, duration, params=default_params, ts=None):
        self.duration = duration
        self.p_1_ = np.exp(-duration * params['T_1_inv'])  # 1 - p_1
        self.p_phi_ = np.exp(-duration * params['T_phi_inv'])  # 1 - p_phi
        if ts is not None:
            self.p_phi_ *= self.lamda(params, ts)
        p_xy = (self.p_1_ * self.p_phi_) ** 0.5
        data = np.diag([1, 1 - self.p_1_, 0, self.p_1_]) + np.diag([p_xy, 0, 0, p_xy])[::-1]
        super().__init__(1, data, name='Idle')

    def lamda(self, params, ts):
        chi = params['chi']
        alpha0 = params['alpha0']
        kappa = params['kappa']
        tstart = ts['start']
        tend = ts['end']
        Dt = ts['Dt']
        two_chi_alpha_Dt = 2 * chi * alpha0 * np.exp(-kappa * Dt)

        # Top half of the integral
        int_term_top = -np.exp(-kappa * tend) / (4 * chi**2 + kappa**2) * \
            (kappa * np.sin(2 * chi * tend) + 2 * chi * np.cos(2 * chi * tend))
        int_term_bot = -np.exp(-kappa * tstart) / (4 * chi**2 + kappa**2) * \
            (kappa * np.sin(2 * chi * tstart) + 2 * chi * np.cos(2 * chi * tstart))

        lamda = np.exp(-two_chi_alpha_Dt * (int_term_top - int_term_bot))
        return lamda


def NoisyHGate(params=default_params, qubit=0):
    if params is None:
        return HGate
    res = Circuit('NoisyH')
    res.append(HGate, [qubit])
    res.append(Depolarization(params['p_axis'] / 4, params['p_plane'] / 2 - params['p_axis'] / 4, params['p_axis'] / 4),
               [qubit])
    return res


def add_idle_noise(circuit, start=None, end=None, params=default_params):
    timing = {q: start for q in circuit.all_qubits}
    time_last_meas = {q: None for q in circuit.all_qubits}
    time_start = {q: None for q in circuit.all_qubits}
    for t, layer in circuit.operations_by_time.items():
        for op in layer.values():
            operation = op['operation']
            for q in op['qubits']:
                if operation.name == 'ND':
                    time_last_meas[q] = t
                t0 = timing[q]
                if t0 is not None and not isinstance(operation,
                                                     State) and (t - t0) > 0:
                    ts = None
                    if time_start[q] is not None:
                        ts = {
                            'start': t0 - time_start[q],
                            'end': t - time_start[q],
                            'Dt': time_start[q] - time_last_meas[q]
                        }
                    circuit.append(IdleGate(t - t0, params=params, ts=ts),
                                   qubits=[q],
                                   time_step=(t0 + t) / 2)
                if isinstance(operation, Measurement):
                    timing[q] = None
                else:
                    timing[q] = t
                if operation.name == 'NoisyH' and time_last_meas[q] is not None:
                    if time_start[q] is None:
                        time_start[q] = t
                    else:
                        time_start[q] = None
                        time_last_meas[q] = None
    if end is not None:
        for q, t0 in timing.items():
            if t0 is not None and t0 < end:
                circuit.append(IdleGate(end - t0, params=params),
                               qubits=[q],
                               time_step=(t0 + end) / 2)


def add_CZ_rotations(circuit, high_freq_group, low_freq_group, time_step=None, angles=None):
    """I will just assume that in the new net-zero gate, the quasi-static flux has a neglected effect. So I changed
    Fang's implementation. Now (angle == pi or None) is the CZ gate. Very small angle is used to add cross-talk.

    Maybe we can add some rotation error
    """

    for q1 in high_freq_group:
        if angles is not None:
            if isinstance(angles, dict):
                gate = Diagonal(2, np.array([1, 1, np.exp(1j * angles[q1]), -np.exp(-1j * angles[q1])]), name='NoisyCZ')
            else:
                gate = Diagonal(2, np.array([1, 1, 1, np.exp(1j * angles)]), name='NoisyCZ')
        else:
            gate = CZGate
        for q2 in low_freq_group:
            if abs(q1[0] - q2[0]) == 1 and abs(q1[1] - q2[1]) == 1:
                circuit.append(gate, [q1, q2], time_step=time_step)


C = TensorNetwork([0, 0, 0, 0], bond_dim=2)
C.add_node('PH', [0], np.ones(2))
NDCompMeas = Channel(1, C, name='ND')


def add_noisy_surface_code(circuit, qubit_coords=None, connections=None, time=None, params=default_params):

    if time is None:
        time = max(circuit.max_time, 0)
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
    for x, y in qubit_coords:
        qubit_groups[qubit_group_name[x % 4, y % 4]].append((x, y))
    if params is None:
        quasi_static_flux = None
        T_g_1Q = 1
        T_g_2Q = 1
        T_C = -1
        tau_m = 1
        tau_d = 4
    else:
        quasi_static_flux = {q: np.random.randn() * params['delta_phi'] for q in qubit_coords}
        T_g_1Q = params['T_g_1Q']
        T_g_2Q = params['T_g_2Q']
        T_C = params.get('T_C', -T_g_1Q)
        tau_m = params['tau_m']
        tau_d = params['tau_d']
        gamma = params['gamma']

    # Time slot A
    time += T_g_1Q / 2
    for group in ['D1', 'D2', 'D3', 'D4', 'X1', 'X2']:
        for qubit in qubit_groups[group]:
            circuit.append(NoisyHGate(params=params), [qubit], time_step=time)
    time += T_g_1Q / 2

    # Time slot 1 ~ 4
    for flux_dance in [('D2', 'X1', 'X2', 'D3', 'D4'),
                       ('D1', 'X1', 'X2', 'D4', 'D3'),
                       ('D1', 'X2', 'X1', 'D4', 'D3'),
                       ('D2', 'X2', 'X1', 'D3', 'D4')]:
        time += T_g_2Q / 2
        for g1, g2 in [flux_dance[0:2], flux_dance[2:4], (flux_dance[4], 'dummy')]:
            add_CZ_rotations(circuit, qubit_groups[g1], qubit_groups[g2], time_step=time, angles=quasi_static_flux)
        time += T_g_2Q / 2

    # cross talk between data qubits and Z-ancilla qubits. added on the same time, so dont add too many idle errors
    for flux_dance in [('D1', 'Z1', 'Z2', 'D4', 'D3'),
                       ('D2', 'Z2', 'Z1', 'D3', 'D4'),
                       ('D2', 'Z1', 'Z2', 'D3', 'D4'),
                       ('D1', 'Z2', 'Z1', 'D4', 'D3')]:
        for g1, g2 in [flux_dance[0:2], flux_dance[2:4], (flux_dance[4], 'dummy')]:
            add_CZ_rotations(circuit, qubit_groups[g1], qubit_groups[g2], time_step=time, angles=gamma)

    # Time slot B
    time += T_g_1Q / 2
    for group in ['D1', 'D2', 'D3', 'D4', 'X1', 'X2']:
        for qubit in qubit_groups[group]:
            circuit.append(NoisyHGate(params=params), [qubit], time_step=time)
    time += T_g_1Q / 2

    # Time slot C
    for group in ['X1', 'X2']:
        for qubit in qubit_groups[group]:
            circuit.append(NDCompMeas, [qubit], time_step=time + tau_m / 2)
    end_time = time + tau_m + tau_d
    time += T_C

    # Time slot D
    time += T_g_1Q / 2
    for group in ['Z1', 'Z2']:
        for qubit in qubit_groups[group]:
            circuit.append(NoisyHGate(params=params), [qubit], time_step=time)
    time += T_g_1Q / 2

    # Time slot 5 ~ 8
    for flux_dance in [('D1', 'Z1', 'Z2', 'D4', 'D3'),
                       ('D2', 'Z2', 'Z1', 'D3', 'D4'),
                       ('D2', 'Z1', 'Z2', 'D3', 'D4'),
                       ('D1', 'Z2', 'Z1', 'D4', 'D3')]:
        time += T_g_2Q / 2
        for g1, g2 in [flux_dance[0:2], flux_dance[2:4], (flux_dance[4], 'dummy')]:
            add_CZ_rotations(circuit, qubit_groups[g1], qubit_groups[g2], time_step=time, angles=quasi_static_flux)
        time += T_g_2Q / 2

    # cross talk between data qubits and X-ancilla qubits. added on the same time, so don't add too many idle errors
    for flux_dance in [('D2', 'X1', 'X2', 'D3', 'D4'),
                       ('D1', 'X1', 'X2', 'D4', 'D3'),
                       ('D1', 'X2', 'X1', 'D4', 'D3'),
                       ('D2', 'X2', 'X1', 'D3', 'D4')]:
        for g1, g2 in [flux_dance[0:2], flux_dance[2:4], (flux_dance[4], 'dummy')]:
            add_CZ_rotations(circuit, qubit_groups[g1], qubit_groups[g2], time_step=time, angles=gamma)

    # Time slot E
    time += T_g_1Q / 2
    for group in ['Z1', 'Z2']:
        for qubit in qubit_groups[group]:
            circuit.append(NoisyHGate(params=params), [qubit], time_step=time)
    time += T_g_1Q / 2

    # Time slot F
    for group in ['Z1', 'Z2']:
        for qubit in qubit_groups[group]:
            circuit.append(NDCompMeas, [qubit], time_step=time + tau_m / 2)

    return end_time
