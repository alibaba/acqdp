import sys
import opt_einsum
from acqdp.tensor_network.local_optimizer import defaultOrderResolver

if sys.version_info < (3, 0):
    sys.stdout.write("Sorry, requires Python 3.x, not Python 2.x\n")
    sys.exit(1)


class OrderFinder:
    """
    :class:`OrderFinder` class is dedicated to finding a contraction scheme corresponding to a given tensor network structure.
        The main method of an :class:`OrderFinder` is `find_order`, which takes a `TensorNetwork` as input, and yields a
        generator of `ContractionScheme`.The base class offers preliminary order finding schemes. For more advanced
        hypergraph-based approach, use :class:`KHPOrderFinder`. For finding contraction schemes with sliced edges,
        use :class:`SlicedOrderFinder`.

    :ivar order_method: 'default' order contracts the tensor one by one by order; 'vertical' order contracts the tensor based
        on the vertical ordering. When the tensor network is from a quantum circuit, the vertical order corresponds to the
        tensor network contraction where the tensors connected to a qubit is first contracted together and then merged to a
        main one according to a given order of the qubits.
    """

    def __init__(self,
                 order_method='default',
                 **kwargs):
        self.order_method = order_method

    def find_order(self, tn, **kwargs):
        """Find a contraction scheme for the input tensor network subject to the constraints given in the
        :class:`OrderFinder`.

        :param tn: A tensor network for which the contraction scheme is to be determined.
        :type tn: :class:`TensorNetwork`
        :yields: A :class:`ContractionScheme` containing the pairwise contraction order, a list of edges to be sliced, and
            optionally the total contraction cost.
        """
        tn = tn._expand_and_delta()
        try:
            if self.order_method == 'default':
                nodes_list = sorted(tn.nodes_by_name)
            elif self.order_method == 'vertical':
                qubit_order = kwargs.get('qubit_order', sorted(set(b[1] for b in tn.nodes_by_name)))
                nodes_list = sorted(tn.nodes_by_name, key=lambda b: (qubit_order.index(b[1]), b[0], b[2:]))
            else:
                raise ValueError("order method not implemented")
        except TypeError:
            nodes_list = sorted(tn.nodes_by_name, key=lambda x: str(x))
        o = []
        if len(nodes_list) > 1:
            k = nodes_list[0]
            for i in range(len(nodes_list) - 1):
                new_k = ('#', i)
                o.append([[nodes_list[i + 1], k], new_k])
                k = new_k
            if len(o) > 0:
                o[-1][-1] = '#'
        res = defaultOrderResolver.order_to_contraction_scheme(tn, o)
        while True:
            yield res


class OptEinsumOrderFinder(OrderFinder):
    """
    :class: `OptEinsumOrderFinder` finds an unsliced contraction scheme based on the built-in method in `opt_einsum`,
    called `opt_einsum.contract_path`.
    :ivar optimize: The argument `optimize` for `opt_einsum.contract_path`.
    """

    def __init__(self,
                 optimize='greedy',
                 **kwargs):
        self.optimize = optimize

    def find_order(self, tn, **kwargs):
        tn = tn._expand_and_delta()
        while True:
            lhs, rhs, shapes = tn.subscripts()
            path, _ = opt_einsum.contract_path(','.join(lhs) + '->' + rhs,
                                               *shapes,
                                               shapes=True,
                                               optimize=self.optimize)
            if len(tn.nodes_by_name) > 1:
                order = defaultOrderResolver.path_to_paired_order([list(tn.nodes_by_name), '#'], path)
            else:
                order = []
            yield defaultOrderResolver.order_to_contraction_scheme(tn, order)


class SlicedOrderFinder(OrderFinder):
    """
    :class: `SlicedOrderFinder` finds a sliced contraction scheme based on unsliced contraction schemes found by its base
        order finder.
    :ivar base_order_finder: The base order finder of the sliced order finder, from which the `SlicedOrderFinder` fetches
        contraction schemes and do slicing on it.
    :ivar slicer: The slicing algorithm acting upon the contraction schemes.
    :ivar num_candidates: Number of unsliced contraction schemes to feed to the slicer at a time. Set to 20 by default.
    """

    def __init__(self,
                 base_order_finder={'order_finder_name': 'khp'},
                 slicer={'slicer_name': 'default'},
                 num_candidates=20,
                 **kwargs):
        self.base_order_finder = base_order_finder
        self.num_candidates = num_candidates
        from acqdp.tensor_network.order_finder import get_order_finder
        from acqdp.tensor_network.slicer import get_slicer
        self.base_order_finder = get_order_finder(**base_order_finder)
        self.slicer = get_slicer(**slicer)

    def find_order(self, tn, **kwargs):
        tn = tn._expand_and_delta()
        order_gen = self.base_order_finder.find_order(graph=tn)
        next(order_gen)
        while True:
            res = self.slicer.slice(tn, order_gen)
            yield res


def get_order_finder(**kwargs):
    from acqdp.tensor_network.kahypar_order_finder import KHPOrderFinder
    order_finders = {
        'khp': KHPOrderFinder,
        'default': OrderFinder,
        'oe': OptEinsumOrderFinder,
        'sliced': SlicedOrderFinder
    }
    order_finder_name = kwargs.get('order_finder_name', 'default')
    return (order_finders[order_finder_name])(
        **kwargs.get('order_finder_params', {}))
