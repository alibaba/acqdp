import json
import numpy as np
import copy
from acqdp.tensor_network.contractor import defaultContractor


class ContractionCost:
    """
    :class:`ContractionCost` is a wrapper for cost quantities related to a tensor network contraction.

    :ivar s: Total number of floating point operations, as a proxy for the time complexity.
    :ivar t: Size of largest intermediate tensor, as a proxy for the space complexity.
    :ivar k: Number of indices sliced, as a proxy for the parallelism
    """

    def __init__(self, s=0, t=0, k=0):
        self.s = s
        self.t = t
        self.k = k

    def __add__(self, other):
        return ContractionCost(self.s + other.s, max(self.t, other.t))

    def __iadd__(self, other):
        self.s = self.s + other.s
        self.t = max(self.t, other.t)
        return self

    def __eq__(self, other):
        return self.s * 2**self.k == other.s * 2**other.k and self.t == other.t

    def __lt__(self, other):
        return self.s * 2**self.k < other.s * 2**other.k

    def __str__(self):
        return "cost = {}, cw = {}, num_slice = {}".format(
            np.log10(float(self.s * 2**self.k)), np.log2(float(self.t)), self.k)


class ContractionScheme:
    """
    :class:`ContractionScheme` is an intermediate representation of a tensor network contraction, containing the pairwise
        contraction orders, the hyperedges to be sliced, and the cost of the contraction. A :class:`ContractionScheme` does
        not contain tensor data.

    :ivar order: The pairwise sequential contraction order.
    :ivar slices: The hyperedges to be sliced for parallelism.
    :ivar cost: :class:`ContractionCost` -- The time / space cost and number of sub-processes of the tensor network
        contraction.
    """

    def __init__(self, order, slices=None, cost=None):
        self.order = order
        if slices is None:
            slices = []
        self.slices = slices
        if cost is None:
            cost = ContractionCost()
        self.cost = cost

    def dump(self, f):
        """Dump the :class:`ContractionScheme` into a json file.

        :param f: The file to dump the :class:`ContractionScheme` to.
        """
        json.dump(
            {
                'contractions': [l for l in self.order],
                'slices': [l for l in self.slices],
                'cost': [int(self.cost.s), int(self.cost.t), int(self.cost.k)]
            },
            f,
            indent=4)

    @classmethod
    def load(cls, f):
        """Load a :class:`ContractionScheme` from an existing file.

        :param f: The file to load the :class:`ContractionScheme` from.

        :returns: :class:`ContractionScheme`
        """
        def tuplize(l):
            if isinstance(l, list):
                return tuple(tuplize(x) for x in l)
            return l
        res = json.load(f)
        return cls([[[tuplize(l0), tuplize(l1)],
                     tuplize(l)] for (l0, l1), l in res['contractions']],
                   [tuplize(l) for l in res['slices']],
                   ContractionCost(*res['cost']))


class OrderCounter:
    @property
    def cnt(self):
        self.k += 1
        return ('#', self.k)

    def __init__(self, order=None):
        self.k = 0
        if order is not None:
            for o in order:
                for k in o[0]:
                    if isinstance(k, tuple) and len(k) == 2 and k[0] == '#':
                        self.k = max(self.k, k[1])

    def __copy__(self):
        oc = OrderCounter([[[('#', self.k)], '#']])
        return oc


class ContractionTask:
    """
    :class:`ContractionTask` is post-compilation intermediate representation for tensor network contraction, with all the
        information necessary to carry out a tensor network contraction, including initial tensor data and stepwise
        instruction of the contraction.
    """

    def __init__(self,
                 output,
                 inputs=None,
                 commands=None,
                 data=None,
                 length=1,
                 shape=None,
                 fix_dict=None,
                 open_edges=None,
                 sub_outputs=None,
                 cnt=0,
                 dtype=complex):
        if commands is None:
            self.commands = []
        else:
            self.commands = commands
        self.output = output
        if inputs is None:
            self.inputs = []
        else:
            self.inputs = inputs
        if data is None:
            self.data = {}
        else:
            self.data = data
        if shape is None:
            self.shape = ()
        else:
            self.shape = shape
        if fix_dict is None:
            fix_dict = {}
        self.fix_dict = fix_dict
        if sub_outputs is None:
            sub_outputs = []
        self.sub_outputs = sub_outputs
        if open_edges is None:
            open_edges = []
        self.open_edges = open_edges
        self.cnt = cnt
        if cnt > 0:
            self.preprocess = False
        length = 1
        for k in self.fix_dict:
            length *= len(k[1])
        self.length = length
        self.dtype = np.dtype(dtype)

    def _merge(self, res_dic):
        if self.length == 1 or len(self.shape) == 0 or self.open_edges is None:
            return sum([res_dic[k] for k in res_dic])
        for i in res_dic:
            res_type = np.array(res_dic[i]).dtype
        lst_id = [i for i in self.open_edges if i not in self.sub_outputs]
        lst = [
            i for i in range(len(self.shape))
            if self.open_edges[i] not in self.sub_outputs
        ] + [self.open_edges.index(i) for i in self.sub_outputs]

        res = np.transpose(
            np.zeros(self.shape, dtype=res_type),
            lst)
        for k in res_dic:
            a = k
            ax = {}
            for l in self.fix_dict:
                if l[0][1] not in self.sub_outputs:
                    ax[l[0][1]] = a % len(l[1])
                    a //= len(l[1])
            ls = []
            for l in lst_id:
                ls.append(ax[l])

            res[tuple(ls)] += res_dic[k]
        res = np.transpose(
            res, [lst.index(i) for i in range(len(self.shape))])  # lst)
        return res

    def update_fix(self, tn):
        """Update the tensor index fixes according to the input tensor network. Use when the contraction task
        corresponds to the same tensor network with different fix configurations. One example of the use case is when
        the contraction task is for an entry of the amplitude of a quantum circuit, whereas the tensor network
        represents the same quantum circuit with a different amplitude entry.

        :param tn: The tensor network to update the fixes to.
        :type tn: :class:`TensorNetwork`.
        """
        new_dict = {}
        for k in self.fix_dict:
            if 'fix_to' in tn.network.nodes[k[0]]:
                c = tn.network.nodes[k[0]]['fix_to']
                new_k = (k[0], tuple(c))
                new_dict[new_k] = self.fix_dict[k]
                new_dict[new_k][0] = list(c)[0]
            else:
                new_dict[k] = self.fix_dict[k]
        self.fix_dict = new_dict
        self.preprocess = False

    def _load_data(self):
        try:
            dilist = sorted(list(self.inputs))
        except TypeError:
            dilist = sorted(list(self.inputs), key=str)
        try:
            ddlist = sorted(list(self.data))
        except TypeError:
            ddlist = sorted(list(self.data), key=str)

        for d in range(len(self.inputs)):
            di = dilist[d]
            dd = ddlist[d]
            if isinstance(self.data[dd], ContractionTask):
                self.inputs[di][0] = (0, np.array(self.data[dd].execute()))
            else:
                self.inputs[di][0] = (0, np.array(self.data[dd]))
        self.preprocess = False
        return self

    def set_data(self, data):
        """Update the data in the contractions. Use when a connectivity of a tensor network remains unaltered but the
        tensor data is changed.

        :param data: a dictionary of tensor data, with keys being their names in the tensor network, and values the
            corresponding tensor data.
        :type data: dict
        """
        self.data = {d: np.array(data[d], dtype=self.dtype) for d in data}

    def cast(self, dtype):
        """Cast the contraction to a different type. All input tensors will be cast to the type, and therefore all steps
        of contraction will be of the type, together with the final result.

        :param dtype: The dtype to cast the computations to.
        :type dtype: `str` or `type`
        """
        self.dtype = np.dtype(dtype)
        self.preprocess = False
        self.set_data(self.data)

    def __getitem__(self, idx):
        """Get a subtask induced by slicing as another :class:`ContractionTask`.

        :param idx: Index of the subtask.
        :type idx: int

        :returns: :class:`ContractionTask` -- The contraction task corresponding to the subtask specified by the index.
        """
        if not self.preprocess:
            self._load_data()
            self.preprocess = True
            defaultContractor._execute(self, cnt=self.cnt)
        self_copy = copy.copy(self)
        for k in self_copy.fix_dict:
            a = idx % len(k[1])
            idx = idx // len(k[1])
            self_copy.fix_dict[k][0] = k[1][a]
        self_copy.commands = self.commands[self.cnt:]
        self_copy.fix_dict = {}
        self_copy.length = 1
        self_copy.cnt = 0
        self_copy.open_edges = None
        self_copy.data = {}
        self_copy.inputs = {}
        return self_copy

    def execute(self, **kwargs):
        """Execute a :class:`ContractionTask`.

        Equivalent to :meth:`Contractor(**kwargs).execute(self)`.
        """
        from acqdp.tensor_network.contractor import Contractor
        return Contractor(**kwargs.get('contractor_params', {})).execute(self)
