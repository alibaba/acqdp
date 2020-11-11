from acqdp.tensor_network.contraction import OrderCounter, ContractionCost
import opt_einsum
import copy
import numpy as np

ORDER_RESOLVER_KWARGS = {'optimal': 8, 'dp': 20, 'thres': 1e8}


class OrderResolver:
    """Interfaces for conveniently converting between different formats of contraction orders. Also serves as a
    converter from non-pairwise contraction orders to pairwise contraction orders.

    :ivar optimal: The maximum number of tensors in a single step of a non-pairwise order, for which brute-force search is
        used to reconstruct the pairwise order.
    :ivar dp: The maximum number of tensors in a single step of a non-pairwise order, for which dynamic programming is used
        to reconstruct the pairwise order.
    :ivar thres: The minimum cost of a non-pairwise contraction order via default path to invoke more advanced search for a
        pairwise order, such as the `dp` or `optimal` methods.
    """

    def __init__(self,
                 optimal=8,
                 dp=20,
                 thres=1e8,
                 **kwargs):
        self.optimal = optimal
        self.dp = dp
        self.thres = thres

    def order_to_contraction_scheme(self, tn, order):
        subscripts = self.order_to_subscripts(tn, order)
        cost = ContractionCost()
        counter = OrderCounter(order)
        new_order = []
        for o, i in zip(order, subscripts):
            try:
                res = self.subscript_to_path(*i)
                new_order += self.path_to_paired_order(o, res[0], counter)
                cost += res[1]
            except IndexError as e:
                print(o, i, res[0], res[1])
                raise e
        from acqdp.tensor_network import ContractionScheme
        return ContractionScheme(new_order, cost=cost)

    def order_to_path(self, tn, order):
        tn_copy = tn.copy()
        tn_copy.fix()
        edges_list = tn_copy.edges_by_name
        nodes_list = tn_copy.nodes_by_name
        lhs = [
            ''.join(
                opt_einsum.get_symbol(edges_list.index(e))
                for e in tn_copy.network.nodes[(0, n)]['edges'])
            for n in nodes_list
        ]
        rhs = ''.join(
            opt_einsum.get_symbol(edges_list.index(e))
            for e in tn_copy.open_edges)
        path = []
        for o in order:
            i, j = nodes_list.index(o[0][0]), nodes_list.index(o[0][1])
            path.append((i, j))
            nodes_list.pop(max(i, j))
            nodes_list.pop(min(i, j))
            nodes_list.append(o[1])
        return ','.join(lhs) + '->' + rhs, path, {
            opt_einsum.get_symbol(edges_list.index(e)): e for e in edges_list
        }

    def order_to_subscripts_tree(self, tn, order):
        from acqdp.tensor_network.contraction_tree import ContractionTree
        tree = ContractionTree.from_order(tn, order)
        return tree.full_subscripts, tree.full_order

    def order_to_subscripts(self, tn, order):
        tn_copy = tn.copy()
        subs = []
        for o in order:
            if o[1] == '#':
                break
            try:
                _, stn = tn_copy.encapsulate(o[0], stn_name=o[1])
                subs.append(stn.subscripts(o[0]))
            except Exception as e:
                print(e)
                raise e
        if len(order) > 0:
            subs.append(tn_copy.subscripts(order[-1][0]))
        return subs

    def print_order(self, tn, order):
        subs, order = self.order_to_subscripts_tree(tn, order)
        steps = []
        for i, sub in enumerate(subs):
            a = len(set(sub[0][0]).difference(sub[0][1]))
            b = len(set(sub[0][1]).difference(sub[0][0]))
            c = len(set(sub[0][1]).intersection(sub[0][0]).difference(sub[1]))
            d = len(set(sub[0][1]).intersection(sub[0][0]).intersection(sub[1]))
            steps.append(
                f'Step {i}: <{a+b+c+d}> {sub[0][0]}[{a} x {c} x {d}] * {sub[0][1]}[{c} x {b} x {d}]'
                f' -> {sub[1]}[{a} x {b} x {d}] \n {order[i]}'
            )
        return '\n'.join(steps)

    def subscript_to_path(self, lhs, rhs, shapes):
        optimize = None
        try:
            path, path_info = opt_einsum.contract_path(','.join(lhs) + '->' + rhs,
                                                       *shapes,
                                                       shapes=True,
                                                       optimize='auto')
        except ValueError as e:
            print(','.join(lhs) + '->' + rhs, shapes)
            raise e
        if len(lhs) > 2 and path_info.opt_cost > self.thres:
            if len(lhs) < self.optimal:
                optimize = 'optimal'
            elif len(lhs) < self.dp:
                optimize = 'dp'
        if optimize is not None:
            path, path_info = opt_einsum.contract_path(','.join(lhs) + '->' + rhs,
                                                       *shapes,
                                                       shapes=True,
                                                       optimize=optimize)
        return path, ContractionCost(path_info.opt_cost,
                                     path_info.largest_intermediate)

    def path_to_paired_order(self, order, path, counter=None):
        lhs, rhs = copy.copy(order[0]), order[1]
        if counter is None:
            counter = OrderCounter()
        new_order = []
        for i in path:
            try:
                if len(i) == 1:
                    return [[[lhs.pop(i[0])], rhs]]
                else:
                    new_order.append([[lhs[i[0]], lhs[i[1]]], counter.cnt])
                    lhs.pop(max(i[0], i[1]))
                    lhs.pop(min(i[0], i[1]))
                    lhs.append(new_order[-1][1])
            except IndexError as e:
                print(order[0], i)
                raise e
        new_order[-1][1] = rhs
        return new_order


defaultOrderResolver = OrderResolver()


class LocalOptimizer:
    """

    :class:`LocalOptimizer` takes a pairwise contraction tree and do local optimization on the tree. Note that a connected
        subgraph of the contraction tree represents a sequence of intermediate steps that take some (potentially intermediate)
        tensors as input, and outputs a later intermediate tensor. This sequence of steps can be optimized based on solely the
        shapes of the input and output tensors associated to the subgraph, without looking at the rest of the contraction
        tree. This allows local reorganization of the tree.

    Each iteration of local optimization consists of two steps: In the first step, the contraction tree is divided into
    non-overlapping subgraphs each up to a given size, with the internal connections in the subgraphs removed. The internal
    connections are then reconstructed in the second phase using accelerated optimum contraction tree finding approach. The
    iterations can be repeated by each time randomly selecting a different patch of subgraph division.

    :ivar size: The maximum number of nodes to be included in one subgraph.
    :ivar resolver: The resolver to help reconstruct the pairwise order from subgraph divisions.

    """

    def __init__(self,
                 size=13,
                 **kwargs):
        self.size = size
        self.resolver = OrderResolver(**kwargs.get('resolver_params', {}))

    def _flatten_order(self, tn, order, offset=0):
        tn_copy = tn.copy()
        tn_copy.fix()
        size_dic = {}
        cnt_offset = 0
        subs = self.resolver.order_to_subscripts(tn_copy, order)
        for i, o in enumerate(order):
            size_dic[o[1]] = len(subs[i][1])
            for j, k in enumerate(o[0]):
                size_dic[k] = len(subs[i][0][j])
        cnt = -1
        while -cnt < len(order):
            rhs_list = {}
            for i, o in enumerate(order[:cnt]):
                rhs_list[o[1]] = (i, o[0])
            o = order[cnt]
            lhs = o[0]
            if len(lhs) >= self.size:
                cnt -= 1
                continue
            # choose the largest tensor
            sizes = {k: size_dic[k] for k in o[0]}
            largest_list = sorted(sizes, key=lambda x: sizes[x])[::-1]
            for l in largest_list:
                if (l not in rhs_list) or (sizes[l] < 7) or (
                        len(rhs_list[l][1]) + len(lhs) > self.size):
                    continue
                if cnt_offset < offset:
                    cnt_offset += 1
                    continue
                else:
                    o[0].remove(l)
                    o[0] += rhs_list[l][1]
                    order.pop(rhs_list[l][0])
                    break
            else:
                cnt -= 1
        return order

    def optimize(self, tn, order, num_iter):
        for _ in range(num_iter):
            new_order = self._flatten_order(tn,
                                            order.order,
                                            offset=np.random.randint(
                                                self.size))
            order = self.resolver.order_to_contraction_scheme(tn, new_order)
        return order
