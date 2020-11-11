import copy
import networkx
import warnings
from collections import OrderedDict
from .tensor_valued import TensorValued, DTYPE
import numpy
import itertools
import opt_einsum
from scipy import sparse
from functools import cmp_to_key
import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TensorNetwork(TensorValued):
    """A :class:`TensorNetwork` is a collection of :class:`Tensor`, with indices from different tensors identified and
    summed up to present a new tensor. An example of tensor network is two 2-tensors with a common edge representing the
    matrix multiplication. A :class:`TensorNetwork` is a graphical representation of an Einstein summation.

    :ivar network: attributed :class:`networkx.Graph` object, representing the attributed hypergraph of the tensor network as
        a tanner graph.
    :ivar dtype: dtype of the :class:`TensorValued` object.
    :ivar open_edges: a list of edge names, indicating the outgoing wires of the tensor network.
    """

    def __init__(self,
                 open_edges=None,
                 dtype: type = DTYPE,
                 bond_dim=None) -> None:
        """Constructor of :class:`TensorNetwork` class."""
        super().__init__(dtype)

        self.network = networkx.Graph()
        self.open_edges = []
        if open_edges is not None:
            for edge in open_edges:
                self.open_edge(edge, bond_dim)
        self.id_count = 0
        self.refresh_signature()

    def contract(self, preset=None, config=None, **kwargs):
        """Evaluate the :class:`TensorNetwork` object as an :class:`numpy.ndarray`.

        :param preset: Select a preset mode. If set to `default`, will contract the tensor network with default ordering and
            no post-processing. If set to `khp`, will invoke advanced KaHyPar-based contraction order finding routine. If set
            to `None`, will read config file or input keyword arguments for order finding specifications.
        :type preset: `"default"`, `"khp"` or `None`.
        :param config: File name for the config file. If preset option is not given and config file is given, the keyword
            arguments will be ignored and the specification will be read from the config file.
        :type config: `str`
        :param kwargs: If neither the preset option nor the config file name is given, the program will read the keyword
            arguments for order finding specifications. See :class:`OrderFinder`, :class:`Compiler`, :class:`Contractor` for
            individual specifications.
        :type kwargs: `dict`

        :returns: :class:`numpy.ndarray`.
        """

        if preset == 'default':
            config = os.path.join(THIS_DIR, 'default_params.json')
        elif preset == 'khp':
            config = os.path.join(THIS_DIR, 'khp_params.json')
        elif preset is not None:
            raise ValueError("Invalid preset options")

        if config is not None:
            with open(config, 'r') as f:
                kwargs = json.load(f)

        return self.compile(self.find_order(**kwargs), **kwargs).execute(**kwargs)

    def find_order(self, input_file=None, output_file=None, **kwargs):
        """Find a contraction scheme of the :class:`TensorNetwork` object. Equivalent to
        `next(get_order_finder(**kwargs).find_order(self))`. See :meth:`OrderFinder.find_order`.

        :param input_file: Input file name. When given, load the contraction scheme from the input file.
        :type input_file: str, optional
        :param output_file: Output file name. When given, the contraction scheme found will be dumped into the file.
        :type output_file: str, optional
        :returns: :class:`ContractionScheme`
        """
        from acqdp.tensor_network import ContractionScheme
        if input_file is not None:
            try:
                with open(input_file, 'r') as f:
                    return ContractionScheme.load(f)
            except FileNotFoundError:
                print('Order file not found. Proceed to order finding procedure')
        from acqdp.tensor_network.order_finder import get_order_finder
        res = next(get_order_finder(**kwargs).find_order(self))
        if output_file is not None:
            with open(output_file, 'w') as f:
                res.dump(f)
                print("Order result saved at " + output_file)
        if hasattr(res, 'cost') and res.cost.t > 2**29:
            warnings.warn(
                "The contraction of this tensor network is likely to exceed the memory constraint of 16GB."
                + " Please proceed with caution, or use alternative order finding methods.",
                ResourceWarning
            )
        return res

    def compile(self, order, **kwargs):
        """Compile a :class:`ContractionScheme` corresponding to the :class:`TensorNetwork` object into a runtime
        executable contraction process as a :class:`ContractionTask`. Equivalent to
        `Compiler(**kwargs.get('compiler_params', {})).compile(self, order)` (See :meth:`Compiler.compile`).

        :returns: :class:`ContractionTask`
        """
        from acqdp.tensor_network.compiler import Compiler
        return Compiler(**kwargs.get('compiler_params', {})).compile(self, order)

    def refresh_signature(self):
        self.signature = hash(numpy.random.uniform())

    @property
    def nodes(self):
        return OrderedDict({n: d for n, d in self.network.nodes(data=True) if n[0] == 0})

    @property
    def edges(self):
        return OrderedDict({n: d for n, d in self.network.nodes(data=True) if n[0] == 1})

    @property
    def nodes_by_name(self):
        return [n[1] for n in self.nodes]

    @property
    def edges_by_name(self):
        return [e[1] for e in self.edges]

    @property
    def shape(self):
        """The common property of all :class:`TensorValued` classes, yielding the shape of the object."""
        return tuple([self.network.nodes[(1, e)]['dim'] for e in self.open_edges])

    def __str__(self):
        nodes_str = "Nodes: \n"
        for node in self.nodes:
            nodes_str += str(node[1]) + ": "\
                + str(self.network.nodes[node]['tensor']) + '\n'\
                + "connected to: "\
                + str([e[1] for e in self.network[node]]) + '\n'
        edges_str = "Edges: \n" + str(self.edges_by_name)
        open_str = "Open edges: " + str(self.open_edges)
        return super().__str__() + "\n"\
            + nodes_str\
            + edges_str + "\n"\
            + open_str

    @property
    def is_valid(self):
        """The common property of all :class:`TensorValued` classes, indicating whether the :class:`TensorValued` object
        is valid or not.

        In every step of a program, all existing :class:`TensorValued` object must be valid, otherwise an exception
        should be thrown out; this property is for double checking that the current :class:`TensorValued` object is
        indeed valid.
        """
        for node in self.nodes:
            shape = [self.network.nodes[e]['dim'] for e in self.network[node]]
            lst = self.network.nodes[node]['tensor'].shape
            if lst is None:
                continue
            lst = list(lst)
            if len(lst) != len(shape):
                return False
            for i in range(len(lst)):
                if (lst[i] is not None)\
                   and (shape[i] is not None)\
                   and (lst[i] != (shape[i])):
                    return False
        return True

    @property
    def is_ready(self):
        """The common property of all :class:`TensorValued` classes, indicating whether the current
        :class:`TensorValued` object is ready for contraction, i.e. whether it semantically represents a tensor with a
        definite value.

        In the process of a program, not all :class:`TensorValued` objects need to be ready; however once the `data`
        property of a certain object is queried, such object must be ready in order to successfully yield an
        :class:`numpy.ndarray` object.
        """
        for node in self.nodes:
            tsr = self.network.nodes[node]['tensor']
            if not tsr.is_ready:
                return False
            shape = tuple([self.network.nodes[e]['dim'] for e in self.network[node]])
            if shape != tsr.shape:
                return False
        return True

    @property
    def closed_edges_by_name(self):
        """Return the set of closed edge names, i.e., edges that do not appear in the open edges.

        :returns: :class:`Set` -- The set of closed edge names.
        """
        return set(self.edges_by_name).difference(self.open_edges)

    def edges_by_name_from_nodes_by_name(self, node_names=None):
        """Return the list of all edges connected to any of the input nodes.

        :param node_names: a set of nodes in this tensor network.
        :type node_names: :class:`List`.

        :returns: :class:`Set` -- the set of edges who is connected to some node in the set `nodes`.
        """
        res = []
        if node_names is None:
            node_names = self.nodes_by_name
        for node_name in node_names:
            res += list(self.network[(0, node_name)])
        return set([e[1] for e in res])

    def closed_edges_by_name_from_nodes_by_name(self, node_names):
        """Return the list of all closed edges connected to any of the input nodes.

        :param node_names: a set of nodes in this tensor network.
        :type node_names: :class:`List`.

        :returns: :class:`Set` -- the set of closed edges who is connected to some node in the set `nodes`.
        """
        edges = self.edges_by_name_from_nodes_by_name(node_names)
        return {edge_name for edge_name in edges if set(self.network[(1, edge_name)]).issubset([(0, n) for n in node_names])
                and edge_name not in self.open_edges}

    def nodes_by_name_from_edges_by_name(self, edge_names):
        """Return the set of all nodes connected to the set of input edges.

        :param edge_names: a list of edges in this tensor network.
        :type edge_names: :class:`List`

        :returns: :class:`Set` -- the set of nodes who are connected to some node in the set `edges`.
        """
        res = set()
        for edge_name in edge_names:
            res |= set(self.network[(1, edge_name)])
        return set([n[1] for n in res])

    # tensor network manipulations

    def add_attribute_to_names(self, name):
        """Add an attribute to all the names for nodes and edges of a tensor network.

        :param name: The attribute to be added to all the names.
        :type name: :class:`Tuple` or hashable
        """
        def comb_name(att, name):
            if not isinstance(att, tuple):
                att = (att,)
            if not isinstance(name, tuple):
                name = (name,)
            return att + name
        self.network = networkx.relabel_nodes(
            self.network, {n: (n[0], comb_name(name, n[1])) for n in self.network.nodes}, False)
        self.open_edges = [comb_name(name, e) for e in self.open_edges]
        for node in self.nodes:
            self.network.nodes[node]['edges'] = [comb_name(name, i) for i in self.network.nodes[node]['edges']]

    def update_dimension(self, dic):
        """Update the dimension of the edges in the tensor network.

        :param dic: the key-value pairs of the edges to be updated. The keys are the name of the edges; the values are the
            corresponding updated bond dimensions. If a value is -1, that edge will be updated as a wildcard-dimensional edge,
            meaning that it will be automatically set a bond dimension when connected to tensor-valued objects with a definite
            dimension.
        :type dic: :class:`Dict`
        :raises: AssertionError -- An error will be raised when the dimension update results in inconsistent bond dimensions.
        """
        for edge_name in dic:
            if (1, edge_name) not in self.network.nodes:
                raise ValueError("Edge {} does not exist".format(edge_name))
            edge_data = self.network.nodes[(1, edge_name)]
            edge_dim = edge_data.get('dim')
            if edge_dim is None:
                edge_data['dim'] = dic[edge_name]
            else:
                if dic[edge_name] is not None:
                    assert edge_dim == dic[edge_name], "{}: {} != {}".format(edge_name,
                                                                             dic[edge_name],
                                                                             edge_dim)

    def add_node(self, node_name=None, edges=None, tensor=None, is_open=False):
        """Add a node into the tensor network.

        :param node_name: The name of the node to be added. If not provided, a new name will be assigned.
        :type node_name: :class:`hashable`
        :param edges: The edges the new tensor node is being connected to.
        :type edges: :class:`List` -- list of edges in the tensor network corresponding to the open edges of the new node. If
            not given, defaults to edges `[0, ..., len(shape)]`
        :param tensor: The value of the newly added tensor.
        :type tensor: :class:`tensor_network.tensor_valued.TensorValued`.
        :param is_open: Whether the new indices associated to the new node will be added to open_edges.
        :type is_open: :class:`bool`
        :returns: hashable -- Name of the node added. If none is given as input, return the one automatically signed.
        :raises: ValueError -- An existing node in the tensor network is being added.
        :raises: IndexError -- A node is added with ambiguous connection to the tensor network. Happens when neither the shape
            of the tensor nor the connecting edges are given.
        :raises: AssertionError -- A mismatch in the bond dimensions.
        """
        from .tensor import Tensor
        if (0, node_name) in self.network.nodes:
            raise ValueError("Node {} already exists in the tensor network".format(node_name))
        if not isinstance(tensor, TensorValued):
            tensor = Tensor(tensor)
        if tensor.dtype == complex:
            self.dtype = complex
        if node_name is None:
            node_name = tensor.identifier
        shape = tensor.shape
        if edges is None:
            if shape is not None:
                edges = range(len(shape))
            else:
                raise IndexError("Connection is ambiguous")
        else:
            edges = list(edges)
        self.network.add_node((0, node_name), edges=edges)
        for i, edge_name in enumerate(edges):
            bond_dim = None
            if shape is not None:
                bond_dim = shape[i]
            self.add_edge(edge_name, bond_dim)
        self.network.add_edges_from([((0, node_name), (1, edge_name)) for edge_name in edges])
        self.update_node(node_name, tensor)
        if is_open:
            for edge in edges:
                self.open_edge(edge)
        self.refresh_signature()
        return node_name

    def pop_node(self, node_name):
        """Pop a tensor node from the tensor network. This operation simply removes the node from the hypergraph.

        :returns: dict -- The dict containing all the information of the popped node, including its tensor and hyperedge
            connections.

        :param node_name: Name of the node to be popped.
        :type node_name: hashable
        """
        pop = self.network.nodes[(0, node_name)]
        self.network.remove_node((0, node_name))
        self.refresh_signature()
        return pop

    def remove_node(self, node_name):
        """Remove a tensor node from the tensor network. The difference between removing a node and popping anode is
        that popping a node does not change the output shape of the tensor network, while removing a node would result
        in a new tensor-valued object, where all edges connected to the removed node are appended to the end of the open
        edges list.

        :returns: dict -- The dict containing all the information of the removed node, including its tensor and hyperedge
            connections.
        :param node_name: Name of the node to be removed.
        :type node_name: hashable
        """
        pop = self.pop_node(node_name)
        self.open_edges += pop['edges']
        return pop

    def update_node(self, node_name, tensor=None):
        """Update an existing node of the tensor network by another tensor-valued object.

        :param node_name: Name of the node to be updated.
        :type node_name: hashable
        :param tensor: The new tensor to be put at the node. If `None`, the tensor must be updated with an actual value later
            in order for the tensor network to be ready for contraction.
        :type tensor: :class:`TensorValued` or `None`
        :raises: AssertionError -- Node update results in a bond dimension mismatch.
        """
        from .tensor import Tensor
        if not isinstance(tensor, TensorValued):
            tensor = Tensor(tensor)
        if tensor.dtype == complex:
            if self.dtype != complex:
                self.dtype = complex
                self.refresh_signature()
        self.network.nodes[(0, node_name)]['tensor'] = tensor
        shape = tensor.shape
        if shape is not None:
            self.update_dimension({edge: shape[i] for i, edge in enumerate(self.network.nodes[(0, node_name)]['edges'])})
        else:
            self.update_dimension({edge: None for edge in self.network.nodes[(0, node_name)]['edges']})

    def add_edge(self, edge_name=None, bond_dim=None):
        """Generate an index which is not in this tensor network before.

        :param edge_name: Name of the edge to be added. Will be automatically assigned if none is given.
        :type edge_name: hashable
        :param bond_dim: Bond dimension of the edge. If set to `None`, the edge bond dimension will be wildcard until set
            otherwise.
        :type bond_dim: `int` or `None`

        :returns: hashable -- Name of the newly added edge.
        """
        if edge_name is None:
            self.id_count += 1
            edge_name = "_" + str(self.id_count - 1)
        if (1, edge_name) not in self.network:
            self.network.add_node((1, edge_name), dim=bond_dim)
        elif bond_dim is not None:
            self.update_dimension({edge_name: bond_dim})
        return edge_name

    def open_edge(self, edge_name=None, bond_dim=None):
        """Append an edge to the list of open edges. If the edge does not exist, it will be created.

        :param edge_name: Name of the edge to be opened.
        :type edge_name: hashable
        :param bond_dim: Bond dimension of the edge. See `add_edge`.
        :type bond_dim: `int`
        """
        if edge_name not in self.edges_by_name:
            self.add_edge(edge_name, bond_dim)
        self.open_edges.append(edge_name)
        self.refresh_signature()

    def close_edge(self, edge_name):
        """Make an edge closed. All appearances of the edge will be removed from the open edges list.

        :param edge_name: Edge to be closed.
        :type edge_name: hashable
        """
        while(edge_name in self.open_edges):
            idx = self.open_edges.index(edge_name)
            self.open_edges.pop(idx)
        self.refresh_signature()

    def fix_edge(self, edge_name, fix_to=0, change_fix=False):
        """Fix an edge to a fixed value. The tensor network structure is not yet modified; this method only puts an
        attribute to the fixed hyperedges. The structure will be changed upon calling the `fix` method.

        :param edge_name: Edge to be fixed to a specific value.
        :type edge_name: hashable
        :param fix_to: the value the edge is fixed to. Should be an integer ranging from 0 to `bond_dim - 1`.
        :type fix_to: `int`
        :param change_fix: Mode of `fix_edge`. If `change_fix` is set to `True`, the edge will be fixed to the given value
            regardless of how it is fixed previously. If set to `False`, the fix will apply together with the previous fix,
            resulting in a zero tensor if the two fixes do not match.
        :type change_fix: `bool`
        """
        if isinstance(fix_to, int):
            fix_to = {fix_to}
        else:
            fix_to = set(fix_to)
        assert fix_to.issubset(range(self.network.nodes[(1, edge_name)]['dim'])
                               ), f"Invalid operation: edge being fixed to invalid value {fix_to}"
        if 'fix_to' in self.network.nodes[(1, edge_name)]:
            if not change_fix:
                self.network.nodes[(1, edge_name)]['fix_to'] = list(set(self.network.nodes[(1, edge_name)]['fix_to']) & fix_to)
            else:
                self.network.nodes[(1, edge_name)]['fix_to'][0] = list(fix_to)[0]
        else:
            self.network.nodes[(1, edge_name)]['fix_to'] = list(fix_to)
        if not change_fix:
            self.refresh_signature()

    def fix_index(self, index, fix_to=0):
        """Fix one outgoing index of the tensor network (see :class:`tensor_network.tensor_valued.TensorValued`)."""
        edge = self.open_edges[index]
        self.fix_edge(edge, fix_to)
        self.open_edges.pop(index)
        self.refresh_signature()
        return self

    def fix(self):
        """Fix the tensor network, by removing all hyperedges attributed to a fixed value.

        This operation changes the structure of the tensor network without changing the value of the tensor network.
        """
        edges = list(self.edges)
        for e in edges:
            if 'fix_to' in self.network.nodes[e]:
                for node in self.nodes_by_name_from_edges_by_name([e[1]]):
                    self.network.nodes[(0, node)]['edges'] = [i for i in self.network.nodes[(0, node)]['edges'] if i != e[1]]
                if e[1] not in self.open_edges:
                    self.network.remove_node(e)
                else:
                    self.network.remove_edges_from([(e, n) for n in self.network[e]])
        return self

    def complete_delta(self):
        """Complete the tensor network by adding necessary delta tensors to the tensor network.

        This is in order to resolve the issue that repeated or new indices appear in the open edges, which is not yet
        recognized by Einstein summation conventions. This operation changes the structure of the tensor network without
        changing its value.
        """
        if '*' in self.nodes_by_name:
            return self
        open_edge_names = {}
        edges = self.edges_by_name_from_nodes_by_name()
        if len(self.nodes) == 0:
            id_node = numpy.array(1)
            self.add_node('*', [], id_node)
        for (i, edge) in enumerate(self.open_edges):
            if edge not in open_edge_names:
                open_edge_names.update({edge: 0})
            else:
                open_edge_names[edge] += 1
                new_node = numpy.eye(self.network.nodes[(1, edge)]['dim'])
                self.add_node(("*", edge, open_edge_names[edge]),
                              [edge, (edge, open_edge_names[edge])], new_node)
                self.open_edges[i] = (edge, open_edge_names[edge])
        for edge in open_edge_names:
            diag = None
            if edge not in edges and open_edge_names[edge] == 0:
                diag = [1] * self.network.nodes[(1, edge)]['dim']
            if 'fix_to' in self.network.nodes[(1, edge)]:
                diag = [
                    1 if i in self.network.nodes[(1, edge)]['fix_to'] else 0
                    for i in range(self.network.nodes[(1, edge)]['dim'])
                ]
                self.network.nodes[(1, edge)].pop('fix_to')
            if diag is not None:
                new_node = numpy.array(diag)
                self.add_node(('*', edge, 0), [edge], new_node)
        return self

    def merge_edges(self, edges, merge_to=None):
        """Merge a list of edges as one single edge by identification.

        :param edges: The list of edges to be merged into one.
        :type edges: :class:`List`
        :param merge_to: The name of the edge all the edges are merged to. If set to `None`, the merged edges will appear
            as `edges[0]`
        :type merge_to: hashable
        """
        dim = None
        for edge_name in edges:
            if 'dim' in self.network.nodes[(1, edge_name)]:
                dim = self.network.nodes[(1, edge_name)]['dim']
        if merge_to is None:
            merge_to = sorted(edges)[-1]
        elif (1, merge_to) not in self.network.nodes:
            self.add_edge(merge_to)
        self.update_dimension({merge_to: dim})
        nodes_list = self.nodes_by_name_from_edges_by_name(edges)
        for edge_name in edges:
            if 'fix_to' in self.network.nodes[(1, edge_name)]:
                self.fix_edge(merge_to, self.network.nodes[(1, edge_name)]['fix_to'])
            nodes = self.network[(1, edge_name)]
            self.network.add_edges_from([(node, (1, merge_to)) for node in nodes])
            if edge_name != merge_to:
                self.network.remove_node((1, edge_name))
            # replace occurrence of edge_name by merge_to
        for node in nodes_list:
            self.network.nodes[(0, node)]['edges'] =\
                [merge_to
                    if i in edges
                    else i
                    for i in self.network.nodes[(0, node)]['edges']]
        self.open_edges =\
            [merge_to
                if i in edges
                else i
                for i in self.open_edges]
        assert all((1, e) in self.network.nodes for e in self.open_edges)
        self.refresh_signature()
        return merge_to

    def rewire(self, node_name, leg, rewire_to=None):
        """Rewire an outgoing edge of a tensor node to another edge in the tensor network.

        :param node_name: Name of the tensor node for whom an outgoing edge is to be rewired.
        :type node_name: hashable
        :param leg: Index of the outgoing edge regarding the tensor node. Should be an integer ranging from `0` to
            rank(t) - 1, t being the corresponding tensor.
        :type leg: `int`
        :param rewire_to: The edge name in the tensor network to redirect the leg to. If `None`, it will be rewired to a newly
            created edge.
        :type rewire_to: hashable
        """
        dim = None
        if self.network.nodes[(0, node_name)]['tensor'].shape is not None:
            dim = self.network.nodes[(0, node_name)]['tensor'].shape[leg]
        edge_name = self.network.nodes[(0, node_name)]['edges'][leg]
        if rewire_to not in self.edges_by_name:
            rewire_to = self.add_edge(rewire_to, dim)
        self.network.nodes[(0, node_name)]['edges'][leg] = rewire_to
        self.network.add_edge((0, node_name), (1, rewire_to))
        if edge_name not in self.network.nodes[(0, node_name)]['edges']:
            self.network.remove_edge((0, node_name), (1, edge_name))
        self.refresh_signature()

    def expand(self, nodes=None, recursive=False):
        """Expand all tensor network hierarchical structure in the tensor network.

        Specifically, when there is a tensor node in the tensor network which is a :class:`TensorNetwork` or
        a :class:`TensorView` object, the tensor nodes in the tensor node are lifted up to the current tensor network level
        with corresponding connections.
        """
        from .tensor_view import TensorView
        if nodes is None:
            def cm(a, b):
                try:
                    return a < b
                except TypeError:
                    try:
                        return a < (b,)
                    except TypeError:
                        try:
                            return (a,) < b
                        except TypeError:
                            return str(a) < str(b)
            nodes = sorted(list(self.nodes_by_name), key=cmp_to_key(cm))
        for node_name in nodes:
            if isinstance(self.network.nodes[(0, node_name)]['tensor'], TensorView):
                self.update_node(node_name, self.nodes_by_name[node_name]['tensor'].expand(recursive))
            if issubclass(type(self.network.nodes[(0, node_name)]['tensor']), TensorNetwork):
                pop = self.pop_node(node_name)
                tsr = pop['tensor'].copy()
                tsr.expand(recursive=True)
                connection = pop['edges']
                tsr.add_attribute_to_names(node_name)
                self.network.add_nodes_from(tsr.network.nodes(data=True))
                self.network.add_edges_from(tsr.network.edges)
                open_edges = tsr.open_edges
                g = networkx.Graph()
                g.add_edges_from([(connection[i], open_edges[i]) for i in range(len(open_edges))])
                for ls in networkx.connected_components(g):
                    merge_to = sorted(list(set(ls).intersection(connection)))[-1]
                    self.merge_edges(list(ls), merge_to)
        self.refresh_signature()
        return self

    def subtn(self, nodes=None, edges=None, open_edges=None):
        if nodes is None:
            if edges is None:
                return self.nodes_by_name, self.closed_edges_by_name, self.copy()
            else:
                nodes = self.nodes_by_name_from_edges_by_name(edges)
        else:
            assert all((0, n) in self.network.nodes for n in nodes), \
                f"Illegal sub tensor network {nodes, [(0, n) in self.network.nodes for n in nodes], self.network.nodes}"
            if edges is None:
                edges = self.closed_edges_by_name_from_nodes_by_name(nodes)
            else:
                assert edges.issubset(self.closed_edges_by_name_from_nodes_by_name(nodes)),\
                    "Illegal sub tensor network"
        all_open_edges = [e for e in self.edges_by_name_from_nodes_by_name(
            nodes).difference(edges) if 'fix_to' not in self.network.nodes[(1, e)]]
        if open_edges is None:
            open_edges = all_open_edges
        else:
            assert(set(open_edges) == set(all_open_edges)), "Illegal sub tensor network"

        stn = TensorNetwork(open_edges=open_edges, dtype=self.dtype)
        for node_name in nodes:
            stn.add_node(node_name, self.network.nodes[(0, node_name)]['edges'], self.network.nodes[(0, node_name)]['tensor'])
        for edge in edges:
            for e in self.network.nodes[(1, edge)]:
                stn.network.nodes[(1, edge)][e] = self.network.nodes[(1, edge)][e]
        return nodes, edges, stn

    def encapsulate(self,
                    nodes=None,
                    edges=None,
                    open_edges=None,
                    stn_name=None):
        nodes, edges, stn = self.subtn(nodes, edges, open_edges)
        for node_name in nodes:
            self.pop_node(node_name)
        self.network.remove_nodes_from([(1, edge_name) for edge_name in edges])
        stn_name = self.add_node(stn_name, stn.open_edges, stn)
        self.refresh_signature()
        return stn_name, stn

    def partial_contract(self,
                         nodes=None,
                         edges=None,
                         open_edges=None,
                         stn_name=None):
        stn_name, stn = self.encapsulate(nodes, edges, open_edges, stn_name)
        self.update_node(stn_name, stn.contract())

    def simplify(self, recursive=True):
        flag = True
        while flag:
            flag = False
            for edge in self.edges:
                found = False
                for n0, n1 in itertools.combinations(self.network[edge], 2):
                    fix = len([e for e in set(self.network[n0]).intersection(
                        set(self.network[n1])) if len(self.network[e]) == 2])
                    sz = max(len(self.network[n0]), len(self.network[n1]))
                    new_sz = len(set(self.network[n0]).union(self.network[n1])) - fix
                    if new_sz <= sz:
                        flag = recursive
                        self.encapsulate(nodes=[n0[1], n1[1]])
                        found = True
                        break
                if found:
                    break

    def _expand_and_delta(self):
        return self.copy().expand(recursive=True).complete_delta()

    @property
    def tanner_graph(self):
        nodes = list(self.nodes)
        edges = [e for e in self.open_edges if 'fix_to' not in self.network.nodes[(1, e)]]
        edges += [e for e in self.closed_edges_by_name if 'fix_to' not in self.network.nodes[(1, e)]]
        x_nodes = []
        y_nodes = []
        for i, node in enumerate(nodes):
            edge_list = [i for i in self.network.nodes[node]['edges'] if i in edges]
            x_nodes += [i + 1] * len(edge_list)
            for edge in edge_list:
                y_nodes.append(edges.index(edge))
        edge_list = [j for j, i in enumerate(edges) if i in self.open_edges]
        y_nodes += edge_list
        x_nodes += [0] * len(edge_list)
        res = sparse.csr_matrix(([True] * len(x_nodes), (x_nodes, y_nodes)), dtype=bool)
        return res, [n[1] for n in nodes]

    def subscripts(self, nodes=None):
        if nodes is None:
            nodes = list(self.nodes_by_name)
        edges = [e for e in self.edges_by_name if 'fix_to' not in self.network.nodes[(1, e)]]
        lhs = []
        shapes = []
        for node in nodes:
            lhs.append(''.join([opt_einsum.get_symbol(edges.index(e))
                                for e in self.network.nodes[(0, node)]['edges'] if e in edges]))
            shapes.append(list(2 if self.network.nodes[(1, e)]['dim'] is None else self.network.nodes[(
                1, e)]['dim'] for e in self.network.nodes[(0, node)]['edges'] if e in edges))
        rhs = ''.join([opt_einsum.get_symbol(edges.index(e)) for e in self.open_edges if e in edges])
        return lhs, rhs, shapes

    def copy(self):
        tn = TensorNetwork(copy.copy(self.open_edges), dtype=self.dtype)
        tn.network = self.network.copy()
        for node in tn.nodes:
            tn.network.nodes[node]['tensor'] = tn.network.nodes[node]['tensor']
        tn.signature = self.signature
        return tn

    def __deepcopy__(self, memo):
        tn = TensorNetwork(copy.deepcopy(self.open_edges), dtype=self.dtype)
        tn.network = copy.deepcopy(self.network)
        for node in tn.nodes:
            tn.update_node(node[1], copy.deepcopy(tn.network.nodes[node]['tensor']))
        tn.signature = self.signature
        return tn

    def cast(self, dtype):
        for node in self.nodes:
            self.network.nodes[node]['tensor'] = self.network.nodes[node]['tensor'].cast(dtype)
        return self
