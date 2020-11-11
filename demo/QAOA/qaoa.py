from acqdp.tensor_network import TensorNetwork
from scipy import optimize
import numpy
import tqdm
import time
import itertools


def XRot(angle):
    return numpy.array([[numpy.cos(angle), 1j * numpy.sin(angle)],
                        [1j * numpy.sin(angle), numpy.cos(angle)]])


def checkinstance(csp):
    """Check if a given instance is valid."""
    if type(csp) != dict:
        print("Instance has to be a dictionary and it has type ",
              type(csp))
        return 1
    try:
        for item in csp:
            if len(item) != len(csp[item].shape):
                stt = "(" + "2," * (len(item) - 1) + "2)"
                print("Label of ", item, " has the shape", csp[item].shape,
                      " Must have shape:", stt)
                return 1
            for num in csp[item].shape:
                if num != 2:
                    stt = "(" + "2," * (len(item) - 1) + "2)"
                    print("Label of ", item, " has the shape",
                          csp[item].shape, " Must have shape:", stt)
                    return 1
        return 0
    except Exception:
        print(" instance must be a dictionary of tuples such as:")
        print(" instance = { (0,) : numpy.array([3,0]),")
        print(" (0, 1) : numpy.array([[3,0],[4,5]])}")
        return 1


class QAOAOptimizer:
    """Quantum variational optimization algorithm for CSP optimization, using the Quantum Approximate Optimization
    Algorithm ansatz."""

    def __init__(self, csp, params=None, num_layers=2):
        """Constructor of :class:`CSPQAOAOptimizer` class."""
        if params is None:
            params = 2 * numpy.pi * numpy.random.rand(2 * num_layers)
        self.set_task(csp, num_layers, params)
        assert checkinstance(self.csp) == 0

    def set_task(self, csp, num_layers, params=None, **kwargs):
        self.csp = {key: numpy.asarray(value) for key, value in csp.items()}
        self.clauses = sorted(list(self.csp))
        self.num_clause = len(csp)
        s = set()
        for clause in self.csp:
            s |= set(clause)
        self.lst_var = sorted(list(s))
        self.params = params
        self.id = hash(tuple(csp)) + hash((num_layers))
        self.num_layers = num_layers
        self.data = {i: [None] for i in range(1, num_layers + 1)}
        self.data.update({-i: [None] for i in range(1, num_layers + 1)})
        for clause in self.clauses:
            self.data.update({(0, clause): [self.csp[clause]]})
            self.data.update({(i, clause): [None] for i in range(1, num_layers + 1)})
            self.data.update({(-i, clause): [None] for i in range(1, num_layers + 1)})
        for key, value in kwargs.items():
            setattr(self, key, value)

    def lightcone(self, qubits, csp, num_layers):
        """Construct simplified tensor network corresponding to all QAOA circuit element acting non-trivially on a set
        of qubits."""
        qubits_set = set(qubits)
        turns = {}
        tn = TensorNetwork(dtype=complex)
        for i in range(num_layers):
            for qubit in qubits_set:
                tn.add_node((2 * i + 1, qubit), [(i, qubit), (i + 1, qubit)], None)
                tn.add_node((-2 * i - 1, qubit), [(-i, qubit), (-i - 1, qubit)], None)
            clauses = []
            new_set = set()
            for clause in csp:
                if set(clause).intersection(qubits_set):
                    clauses.append(clause)
                    new_set |= set(clause)
                    tn.add_node((2 * i + 2, clause), [(i + 1, q) for q in clause], None)
                    tn.add_node((-2 * i - 2, clause), [(-i - 1, q) for q in clause], None)
            turns.update({qubit: i + 1 for qubit in new_set.difference(qubits_set)})
            qubits_set |= new_set
        for qubit in turns:
            tn.merge_edges([(turns[qubit], qubit), (-turns[qubit], qubit)], merge_to=(0, qubit))
        tn.update_dimension({e: 2 for e in tn.edges_by_name})
        return tn, qubits_set

    def decorate(self, params=None):
        """Assign specific values to the relavant tensor networks (specified by `tn.dict`) according to the input
        paramter values."""
        if params is None:
            params = self.params
        betas = params[:self.num_layers]
        gammas = params[self.num_layers:]
        for i in range(1, self.num_layers + 1):
            self.data[i][0] = XRot(betas[-i])
            self.data[-i][0] = XRot(-betas[-i])
            for clause in self.clauses:
                self.data[(i, clause)][0] = numpy.exp(1j * gammas[-i] * self.csp[clause])
                self.data[(-i, clause)][0] = numpy.exp(-1j * gammas[-i] * self.csp[clause])
        for tsk in self.query_dict:
            self.query_dict[tsk].set_data(
                {i: self.data_dict[tsk][i][0] for i in self.data_dict[tsk]})

    def preprocess_clause(self, clause, **kwargs):
        tn, set_qubits = self.lightcone(clause, self.csp, self.num_layers)
        multiplier = kwargs.get('multiplier', 1)
        multiplier *= 2 ** (-len(set_qubits))
        tn.add_node((0, clause), [(0, i) for i in clause], None)
        task = tn.compile(tn.find_order(**kwargs), **kwargs)
        dic = {}
        for k in tn.nodes_by_name:
            if k[0] == 0:
                dic[k] = [multiplier * numpy.array(self.data[(0, clause)][0])]
            elif k[0] % 2 == 0:
                dic[k] = self.data[(k[0] // 2, k[1])]
            else:
                dic[k] = self.data[(k[0] + (1 if k[0] > 0 else -1)) // 2]
        return {clause: task}, {clause: dic}

    def preprocess(self, **kwargs):
        """Preprocessing for calculating the energy value of the QAOA circuit."""
        time_start = time.time()
        print("Preprocessing for energy query...")

        self.query_dict = {}
        self.data_dict = {}
        self.clauses = kwargs.get('clauses', self.clauses)
        for clause in tqdm.tqdm(self.clauses):
            m = 1
            if isinstance(clause, dict):
                m = clause['weight']
                clause = clause['clause']
            a, b = self.preprocess_clause(clause, multiplier=m, **kwargs)
            self.query_dict.update(a)
            self.data_dict.update(b)
        print("Preprocessing time for queries: {}".format(time.time() - time_start))

    def optimize(self, method=None, init_value=None, **kwargs):
        """Optimizing over the parameters with respect to the total energy function."""
        if init_value is None:
            init_value = self.params
        elif init_value == 'zeros':
            init_value = numpy.zeros(len(self.params))
        else:
            init_value = numpy.reshape(numpy.array(init_value),
                                       (len(self.params),))
        res = optimize.minimize(lambda x: numpy.real(self.query(x)),
                                init_value,
                                method=method,
                                tol=1.0e-2,
                                options={
                                    'disp': False,
                                    'maxiter': kwargs.get('num_calls', 100)
        })
        params = res.x
        value = res.fun
        self.params = params
        return value, params

    def query(self, params=None, noise_config=None, clauses_list=None, **kwargs):
        """Querying of the total energy corresponding to a specific set of values of the parameters.

        If `None`, the internal parameter values will be used.
        """

        if params is None:
            params = self.params
        self.decorate(params)
        res = []
        for i in tqdm.tqdm(self.query_dict):
            res.append(self.query_dict[i].execute(**kwargs))
        res = sum(res)
        print("E({}) = {}".format(list(params), res))
        return res

    def energy(self, assignment):
        e = 0
        for i in self.csp:
            e += self.csp[i][tuple(assignment[k] for k in i)]
        return e

    def optimum(self):
        x = min(itertools.product([0, 1], repeat=len(self.lst_var)),
                key=lambda a: self.energy(a))
        return self.energy(x), x
