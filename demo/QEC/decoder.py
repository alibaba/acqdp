import numpy as np
import tqdm
import json
from demo.QEC.surface_code import surface_code_tensor_network


def computeErrorRate(params):
    tn = surface_code_tensor_network(num_layers=2, params=params)
    e_ro = params.get('e_ro', 0.01)
    butterfly = np.array([[1 - e_ro, e_ro], [e_ro, 1 - e_ro]])
    with open('acqdp/tensor_network/khp_params.json', 'r') as f:
        kwargs = json.load(f)
    order = tn.find_order(input='task.json', output='task.json', **kwargs)
    task = tn.compile(order, **kwargs)
    # task.set_data({node: tn.network.nodes[(0, node)]['tensor'].contract() for node in tn.nodes_by_name})
    res = task.execute(**kwargs)

    alphabet = 'abABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabet_u = 'cdefghijklmnopqrstuvwxyz'
    # convert from Choi matrices to PTM
    Pauli = np.array([[[[1, 0], [0, 1]], [[0, 1], [1j, 0]]],
                      [[[0, 1], [-1j, 0]], [[1, 0], [0, -1]]]])
    res = np.einsum(alphabet + ',' + 'aAcd->cbdB' + alphabet[4:], res, Pauli)
    res = np.einsum(alphabet + ',' + 'bBcd->cdaA' + alphabet[4:], res,
                    np.conj(Pauli))
    res = np.real(res)

    # Merge with readout error
    for k in tqdm.tqdm(range(16)):
        script = alphabet + ',' + alphabet[k + 4] + alphabet_u[k] + '->'
        output = alphabet[:k + 4] + alphabet_u[k] + alphabet[k + 5:]
        res = np.einsum(script + output, res, butterfly)

    res = np.reshape(res, (4, 4, 2**24))

    # Maximum likelihood decoding
    sk = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1,
                                                                  1]])
    aa = np.argmax(np.einsum('aab,ac->cb', res, sk), axis=0)
    bb = np.zeros((2**24, 4))
    bb[np.arange(2**24)] = sk[aa]
    res = np.einsum('abc,ca->abc', res, bb)
    lk = np.sum(res, axis=2) * 512
    return lk


params = {
    'T_1_inv': 1 / 30000.0,
    'T_phi_inv': 1 / 60000.0,
    'p_axis': 1e-4,
    'p_plane': 5e-4,
    'delta_phi': 0.0,
    'T_g_1Q': 20.0,
    'T_g_2Q': 40.0,
    'tau_m': 300.0,
    'tau_d': 300.0,
    'gamma': 0.12,
    'alpha0': 4,
    'kappa': 1 / 250,
    'chi': 1.3 * 1e-3,
    'xsplit': 2,
    'butterfly': np.array([[1 - 0.0015, 0.0015], [0.0015, 1 - 0.0015]])
}

params['gamma'] = 0.12
b = computeErrorRate(params)
params['gamma'] = 0
a = computeErrorRate(params)
print(a, b, a - b)
