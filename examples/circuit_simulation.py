import numpy as np
import time
import math
from acqdp import circuit
from acqdp.tensor_network import ContractionScheme, TensorNetwork
from datetime import timedelta
import json
import argparse
import os

t = TensorNetwork(open_edges=[0, 1, 1, 0])
t.add_node(0, [0, 1], np.array([[1, 1j], [1j, np.exp(np.pi * 1j / 6)]]))
ISWAP_CZ = circuit.Unitary(2, t, "FSim", True)


def GRCS(f, in_state=0, simplify=False):
    with open(f) as fin:
        n = int(fin.readline())
        c = circuit.Circuit()
        if in_state is not None:
            for qubit in range(n):
                c.append(circuit.CompState[(in_state >> (n - qubit - 1)) & 1],
                         [qubit], -1)
        gate_table = {
            'h':
                circuit.HGate,
            'x_1_2':
                circuit.Unitary(1,
                                np.array([[1 / np.sqrt(2), -1j / np.sqrt(2)],
                                          [-1j / np.sqrt(2), 1 / np.sqrt(2)]]),
                                name='X_1_2'),
            'y_1_2':
                circuit.Unitary(1,
                                np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)],
                                          [1 / np.sqrt(2), 1 / np.sqrt(2)]]),
                                name='Y_1_2'),
            'hz_1_2':
                circuit.Unitary(
                    1,
                    np.array([[1 / np.sqrt(2), -np.sqrt(1j) / np.sqrt(2)],
                              [np.sqrt(-1j) / np.sqrt(2), 1 / np.sqrt(2)]]),
                    name='W_1_2'),
            'cz':
                circuit.CZGate,
            't':
                circuit.Diagonal(1, np.array([1, np.exp(1j * np.pi / 4)])),
            'is':
                circuit.Diagonal(2, np.array([1, 1j, 1j, 1])) | circuit.SWAPGate
        }

        size_table = {
            'h': 1,
            'cz': 2,
            't': 1,
            'x_1_2': 1,
            'y_1_2': 1,
            'hz_1_2': 1,
            'is': 2,
            'rz': 1,
            'fs': 2
        }

        for line in fin:
            words = line.split()
            layer = int(words[0])
            target = list(
                int(x) for x in words[2:2 + size_table[words[1].lower()]])
            params = words[2 + size_table[words[1].lower()]:]
            if not params:
                c.append(gate_table[words[1].lower()], target, layer)
            elif len(params) == 1:
                c.append(
                    circuit.Diagonal(
                        1,
                        np.array([1, np.exp(1j * float(params[0]))]),
                        name='R_Z({})'.format(params[0])), target, layer)
            elif len(params) == 2:
                c.append(
                    circuit.Unitary(
                        2,
                        np.array([[1, 0, 0, 0],
                                  [
                                      0,
                                      math.cos(float(params[0])),
                                      -math.sin(float(params[0])) * 1j, 0
                        ],
                            [
                                      0, -math.sin(float(params[0])) * 1j,
                                      math.cos(float(params[0])), 0
                        ],
                            [
                                      0, 0, 0,
                                      math.cos(-float(params[1])) + math.sin(-float(params[1])) * 1j
                        ]]),
                        name='FSim'), target, layer)
        if simplify:
            for k in c.operations_by_name:
                if c.operations_by_name[k][
                        'time_step'] == 2 or c.operations_by_name[k][
                            'time_step'] == c.max_time - 2:
                    c.operations_by_name[k]['operation'] = ISWAP_CZ
        return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate circuits with tensor network contraction.')
    parser.add_argument('circuit_file', help='the ciruit file (in .qsim format) to be simulated')
    parser.add_argument('-o', '--load-order', metavar='order_file', help='load a contraction order from a file')
    parser.add_argument('-s', '--save-order', metavar='order_file', help='save the contraction order to a file')
    parser.add_argument(
        '-a',
        '--num-amplitudes',
        metavar='N_a',
        default=1,
        type=int,
        help='number of amplitudes that would need to be sampled (used only to calculate the projected running time)')

    args = parser.parse_args()

    start_time_TZ = time.time()

    c = GRCS(args.circuit_file, simplify=False)
    n = len(c.all_qubits)
    tn = c.tensor_pure
    tn.cast(np.complex64)
    tn.expand(recursive=True)

    open_indices = [0, 1, 2, 3, 4, 5]
    for i in range(n):
        if i not in open_indices:
            tn.fix_edge(tn.open_edges[i], 0)
    tn.open_edges = [tn.open_edges[i] for i in open_indices]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_dir, 'khp_params.json'), 'r') as f:
        kwargs = json.load(f)
    if args.load_order is not None:
        print(f'Loading order file {args.load_order}\n')
        with open(args.load_order, 'r') as f:
            order = ContractionScheme.load(f)
    else:
        order = tn.find_order(**kwargs)
    print(order.cost)
    if args.save_order is not None:
        print(f'Saving order file {args.save_order}\n')
        with open(args.save_order, 'w') as f:
            ContractionScheme.dump(order, f)

    tsk = tn.compile(order, **kwargs)
    print("Number of subtasks per batch --- %d ---" % (tsk.length))
    pp_time_TZ = time.time()
    compile_time = time.time()
    print("TaiZhang Preprocessing Time --- %s seconds ---" % (pp_time_TZ - start_time_TZ))
    start_time = time.time()
    results = 0
    num_samps = 5
    tsk.cast('complex64')
    for i in range(num_samps):
        res = tsk[i].execute(**kwargs)
        results += res
    compute_time = time.time()
    print(results)
    tm = timedelta(seconds=args.num_amplitudes * (compute_time - start_time) * tsk.length / num_samps / 27648)
    print("Compute Time       --- %s seconds ---" % (compute_time - start_time))
    print(f'Projected Running Time  --- {tm} ---')
