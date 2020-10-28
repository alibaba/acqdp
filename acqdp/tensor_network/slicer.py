from acqdp.tensor_network.local_optimizer import defaultOrderResolver, LocalOptimizer
from acqdp.tensor_network.contraction import ContractionScheme
from multiprocessing import Pool
import copy
import numpy


class Slicer:
    """
    :class:`Slicer` finds slicing of an unsliced contraction scheme when called by the :class:`SlicedOrderFinder`.

    :ivar num_iter_before: Number of iterations of local optimization before slicing.
    :ivar num_iter_before: Number of iterations of local optimization in the middle of slicing.
    :ivar num_iter_before: Number of iterations of local optimization after slicing.
    :ivar max_num_slice: Maxmimum number of edges to be sliced. If set to -1, the constraint will be ignored.
    :ivar num_threads: Number of threads for multi-processing.
    :ivar slice_thres: Automatically slice edges that introduce an overhed below this threshold. Set to 0.02 by default.
    """

    def __init__(self,
                 num_iter_before=0,
                 num_iter_middle=20,
                 num_iter_after=100,
                 max_tw=29,
                 max_num_slice=-1,
                 num_threads=28,
                 slice_thres=.02,
                 **kwargs):
        self.num_iter_before = num_iter_before
        self.num_iter_middle = num_iter_middle
        self.num_iter_after = num_iter_after
        self.max_tw = max_tw
        self.max_num_slice = max_num_slice
        self.num_threads = num_threads
        self.slice_thres = slice_thres
        self.local_optimizer = LocalOptimizer(
            **kwargs.get('local_optimizer_params', {}))
        self.num_suc_candidates = kwargs.get('num_suc_candidates', 10)

    def _slice(self, tn, orders, num_process=0):
        tn = tn.copy()
        tnc = tn
        while True:
            try:
                tn = tnc.copy()
                y = min(orders, key=lambda a: (a.cost, orders.index(a)))
                slice_edges = []
                print(f'Process {num_process} initial cost: {y.cost}', flush=True)
                y = self.local_optimizer.optimize(
                    tn, y, self.num_iter_before)
                while y.cost.t > 2**self.max_tw:
                    y = self.local_optimizer.optimize(
                        tn, y, self.num_iter_middle)
                    k, order = self._biggest_weight_edge(tn, y.order)
                    slice_edges += k
                    for a in k:
                        tn.fix_edge(a)
                    tn.fix()
                    y = defaultOrderResolver.order_to_contraction_scheme(tn, order)
                    y.cost.k = len(slice_edges)
                    if len(slice_edges) >= self.max_num_slice:
                        break
                    if numpy.log2(float(y.cost.t)) - self.max_tw + len(
                            slice_edges) >= self.max_num_slice + 2:  # early termination
                        break
                new_y = self.local_optimizer.optimize(
                    tn, y, self.num_iter_after)
                new_y.cost.k = len(slice_edges)
                if new_y.cost.t <= 2**self.max_tw:
                    y = new_y
                if y.cost.t <= 2**self.max_tw:
                    print(f'Process {num_process} succeeded with {y.cost}',
                          flush=True)
                    return ContractionScheme(y.order, slice_edges, cost=y.cost)
                else:
                    return None
            except KeyboardInterrupt:
                return None

    def _biggest_weight_edge(self, tn, order):
        """Find an edge or a list of edges, slicing of which introduces an overhead each that is below a threshold given
        by self.slice_thres, or a minimal overhead if self.slice_thres is unattainable.

        The method enumerates all edges that appear frequently on the stem of the contraction tree. It tries to
        introduce as minimal overhead as possible by flipping branches on the stem while trying to slice the edges.
        """
        tn_copy = tn.copy()
        tn_copy.fix()
        nodes_names = list(tn_copy.nodes_by_name)
        from acqdp.tensor_network.undirected_contraction_tree import UndirectedContractionTree
        eq, path, eedd = defaultOrderResolver.order_to_path(tn_copy, order)
        uct = UndirectedContractionTree(eq, path)
        se = set()
        edges_dic = {}
        ss = []
        for i in range(len(uct.stem) - 1):
            new_se = uct.open_subscripts_at_edge(
                uct.graph.nodes[uct.stem[i]]['parent'], uct.stem[i])
            for a in new_se.difference(se):
                edges_dic[a] = i
            for a in se.difference(new_se):
                ss.append((a, edges_dic[a], i))
            se = new_se
        ss = sorted(ss, key=lambda x: x[1] - x[2])[:10]
        ss_dic = {}
        c = uct.cost
        slice_edges = []
        for s in ss:
            uct_copy = copy.deepcopy(uct)
            for v in range(uct_copy.n):
                uct_copy.graph.nodes[v]['subscripts'] = uct_copy.graph.nodes[v][
                    'subscripts'].difference({s[0]})
            for u, v in uct_copy.graph.edges:
                uct_copy.graph[u][v].clear()
                uct_copy.graph[v][u].clear()
                uct_copy.preprocess_edge(u, v)
                uct_copy.preprocess_edge(v, u)
            uct_copy.compute_root_cost()
            for v in range(uct_copy.n, uct_copy.n * 2 - 2):
                uct_copy.compute_node_cost(v)
            res = (uct_copy.cost, s[1], s[2], uct_copy.get_path())
            i = s[1]

            curr_cost = res[0]
            while i > 5:
                ii = i
                while ii > 3:
                    for k in range(3, ii)[::-1]:
                        for l in range(k, ii):
                            uct_copy.switch_branches(l)
                        if uct_copy.cost <= curr_cost:
                            curr_cost = uct_copy.cost
                            ii -= 1
                            break
                        else:
                            for l in range(k, ii)[::-1]:
                                uct_copy.switch_branches(l)
                    else:
                        break

                k = ii
                sss = ii + 1
                while k > 5:
                    uct_copy.switch_branches(k)
                    if uct_copy.cost <= curr_cost:
                        curr_cost = uct_copy.cost
                        sss = k
                    k -= 1
                ii = sss
                for k in range(6, ii):
                    uct_copy.switch_branches(k)
                if ii >= i:
                    break
                i = ii
            res = (curr_cost, i, res[2])
            j = res[2]
            while j < len(uct_copy.stem) - 5:
                jj = j
                while jj < len(uct_copy.stem) - 3:
                    for k in range(jj + 1, len(uct_copy.stem) - 1):
                        for l in range(jj, k)[::-1]:
                            uct_copy.switch_branches(l)
                        ucost = uct_copy.cost
                        if ucost <= curr_cost:
                            curr_cost = ucost
                            jj += 1
                            break
                        else:
                            for l in range(jj, k):
                                uct_copy.switch_branches(l)
                    else:
                        break
                k = jj
                sss = jj - 1
                while k < len(uct_copy.stem) - 5:
                    uct_copy.switch_branches(k)
                    ucost = uct_copy.cost
                    if ucost <= curr_cost:
                        curr_cost = uct_copy.cost
                        sss = k
                    k += 1
                jj = sss
                for k in range(jj + 1, len(uct_copy.stem) - 5)[::-1]:
                    uct_copy.switch_branches(k)
                if jj <= j:
                    break
                j = jj
            if curr_cost < (1 + self.slice_thres) / 2 * c:
                slice_edges.append(eedd[s[0]])
                c = curr_cost
                uct = uct_copy
            else:
                ss_dic[s[0]] = (curr_cost, res[1], j, uct_copy.get_path())
        if len(slice_edges) > 0:
            pp = uct.get_path()
        else:
            kk = sorted(ss_dic, key=lambda x: ss_dic[x][0])[0]
            slice_edges = [eedd[kk]]
            pp = ss_dic[kk][-1]
        new_order = []
        for i, p in enumerate(pp):
            new_order.append([[nodes_names[p[0]], nodes_names[p[1]]], order[i][1]])
            nodes_names.pop(max(p[0], p[1]))
            nodes_names.pop(min(p[0], p[1]))
            nodes_names.append(order[i][1])
        return slice_edges, new_order

    def slice(self, tn, order_gen):
        orders = [next(order_gen) for _ in range(self.num_suc_candidates)]
        return self._slice(tn, orders)


def mpwrapper(slicer, tn, orders, num_process):
    return slicer._slice(tn, orders, num_process)


class MPSlicer(Slicer):
    """Multi-processing slicing, by concurrently trying different slicing routes."""

    def slice(self, tn, order_gen):
        candidates = []
        while len(candidates) <= self.num_suc_candidates:
            with Pool(self.num_threads) as p:
                lk = list((self, tn, [next(order_gen)], num_process) for num_process in range(self.num_threads))
                new_candidates = p.starmap(mpwrapper, lk)
            candidates += [i for i in new_candidates if i is not None]
            print("Num of candidates now: {}".format(len(candidates)))
        res = min(candidates, key=lambda x: (x.cost, candidates.index(x)))
        return res


def get_slicer(**kwargs):
    slicers = {'default': Slicer, 'mp': MPSlicer}
    slicer_name = kwargs.get('slicer_name', 'default')
    return (slicers[slicer_name])(**kwargs.get('slicer_params', {}))
