import kahypar
import numpy
import itertools
import copy
from os.path import join, abspath, dirname
import cma
import os
import tempfile
from multiprocessing import Pool

from acqdp.tensor_network.local_optimizer import OrderResolver
from acqdp.tensor_network.contraction import ContractionScheme, OrderCounter
from acqdp.tensor_network.order_finder import OrderFinder

CMA_TEMP_DIR = tempfile.TemporaryDirectory()
CMA_FILE_PREFIX = CMA_TEMP_DIR.name + os.sep

KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)), 'kahypar_profiles')

modes = ['direct', 'recursive']
objectives = ['cut', 'km1']
queryOrderResolver = OrderResolver(optimal=9, dp=9)


def query(order_finder,
          s,
          eps,
          graph_copy,
          seed=numpy.random.randint(2**31 - 1),
          counter=None,
          objective=0,
          cutoff=25):
    K = s // 2 + 3
    mode = s % 2
    tg, nodes = graph_copy.tanner_graph
    order = order_finder._order_by_params(numpy.array(tg.todense()),
                                          copy.copy(nodes),
                                          K=K,
                                          eps=eps,
                                          mode=mode,
                                          objective=0,
                                          cutoff=cutoff,
                                          seed=seed,
                                          counter=counter)
    res = queryOrderResolver.order_to_contraction_scheme(graph_copy, order)
    return res


class KHPOrderFinder(OrderFinder):
    """
    :class:`KHPOrderFinder` utilizes hypergraph decomposition and cma-es opmization scheme for finding contraction orders for
        intermediate-size tensor networks.

    :ivar num_threads: Number of threads to concurrently search for the best order.
    :ivar num_iters: Number of iterations for cma optimization.
    :ivar num_cmas: Number of times cma-es optimization scheme is called for exploring better parameter combinations.
    :ivar cma_args: Arguments for cma-es optimization scheme. See the documentation of cma for more details
    :ivar eps: In case a good parameter combination is already known, inputting a eps would skip the cma optimization scheme
        and directly yield orders found by hypergraph decomposition.
    """

    def __init__(
            self,
            num_threads=28,
            num_iters=50,
            num_cmas=1,
            cma_args={
                'bounds': [0.5, 0.5, 0.5],
                'std_dev': 0.2,
                'kwargs': {
                    'seed': 1175,
                    'bounds': [[0.01, 0, 0], [1, 1, 1]],
                    'tolfun': 1e-2,
                    'verb_filenameprefix': CMA_FILE_PREFIX
                },
            },
            eps=None,
            **kwargs):
        self.num_threads = num_threads
        self.num_iters = num_iters
        self.num_cmas = num_cmas
        self.eps = eps
        self.cma_args = cma_args

    def _simplify(self, graph):
        order = []
        counter = OrderCounter()
        flag = True
        while flag:
            flag = False
            for edge in graph.edges:
                found = False
                for n0, n1 in itertools.combinations(graph.network[edge], 2):
                    s0, s1 = set(graph.network[n0]), set(graph.network[n1])
                    fix = len([
                        e for e in s0.intersection(s1)
                        if len(graph.network[e]) == 2
                    ])
                    sz = max(len(s0), len(s1))
                    new_sz = len(s0.union(s1)) - fix
                    if new_sz <= sz:
                        flag = True
                        new_name = counter.cnt
                        new_name, _ = graph.encapsulate(nodes=[n0[1], n1[1]],
                                                        stn_name=new_name)
                        order.append([[n0[1], n1[1]], new_name])
                        found = True
                        break
                if found:
                    break
        return order, counter

    def find_order(self, graph):
        graph_copy = graph._expand_and_delta()
        graph_copy.fix()
        init_order, counter = self._simplify(graph_copy)
        if self.eps is None:
            seed_f = numpy.random.uniform(0, 1)

            def probe(x):
                eps = [x[0], 0.95 + 0.049 * x[1]]
                with Pool(self.num_threads) as p:
                    reses = p.starmap(
                        query,
                        [(self, s, eps, graph_copy, int(
                            (2**31 - 1) * seed), copy.copy(counter))
                         for s in range(self.num_threads)
                         for seed in x[2:]])
                res = min(reses, key=lambda x: (x.cost, reses.index(x)))
                return res

            curr_y = None
            for _ in range(self.num_cmas):
                seed_f = numpy.random.uniform(0, 1)
                self.cma_args['kwargs']['seed'] = numpy.random.randint(2**31 - 1)

                def probe_value(x):
                    res = probe([x[0], x[1], seed_f])
                    return numpy.log10(float(res.cost.s))

                es = cma.CMAEvolutionStrategy(
                    self.cma_args['bounds'], self.cma_args['std_dev'],
                    self.cma_args['kwargs']).optimize(probe_value,
                                                      iterations=self.num_iters)
                x = es.result[0]
                y = es.result[1]
                if curr_y is None or curr_y > y:
                    self.eps = [x[0], 0.95 + 0.049 * x[1]]
                    curr_y = y
            res = probe(x)
            yield ContractionScheme(init_order + res.order, cost=res.cost)
        while True:
            reses = [
                query(self,
                      s,
                      self.eps,
                      graph_copy,
                      seed=numpy.random.randint(2**31 - 1),
                      counter=copy.copy(counter)) for s in range(10)
            ]
            res = min(reses, key=lambda x: (x.cost, reses.index(x)))
            yield ContractionScheme(init_order + res.order, cost=res.cost)

    def _order_by_params(self, graph, nodes, **kwargs):
        graph = self._refresh(graph)
        counter = kwargs.get('counter')

        K = int(kwargs.get('K'))
        eps = kwargs.get('eps', None)
        top = kwargs.get('top', True)
        kwargs_c = copy.copy(kwargs)
        kwargs_c['eps'] = eps[0]
        if not top:
            kwargs_c['K'] = 2
            kwargs_c['eps'] = eps[1]
        context = self._set_context(**kwargs_c)
        cutoff = kwargs.get('cutoff')
        hypergraph, _, edges = self._init_hypergraph_tanner(graph, kwargs_c['K'])
        if not edges:
            return [[nodes, '#']]
        kahypar.partition(hypergraph, context)
        partitions_names = [[] for _ in range(K)]
        for i, n in list(enumerate(nodes)):
            partitions_names[hypergraph.blockID(i)].append(n)
        if any(len(part) == len(nodes) for part in partitions_names):
            return [[nodes, '#']]
        order = []
        all_names = []
        for (i, partite_nodes) in enumerate(
                sorted(partitions_names, key=lambda x: -len(x))):
            if len(partite_nodes) == 0:
                continue
            elif len(partite_nodes) > 1:
                new_stn = counter.cnt
                all_names.append(new_stn)
                new_g, graph, nodes = self._sub(graph, nodes, partite_nodes,
                                                new_stn)
                assert graph.shape[0] == len(nodes) + 1, (graph.shape[0],
                                                          len(nodes))
                kwargs['top'] = False
                if len(partite_nodes) > cutoff:
                    new_order = self._order_by_params(new_g, partite_nodes,
                                                      **kwargs)
                    new_order[-1][1] = new_stn
                else:
                    new_order = [[partite_nodes, new_stn]]
                order += new_order
            else:
                all_names.append(partite_nodes[0])
        order.append([all_names, '#'])
        return order

    def _set_context(self, **kwargs):
        mode = modes[int(kwargs.get('mode'))]
        objective = objectives[int(kwargs.get('objective'))]
        K = int(kwargs.get('K'))
        eps = kwargs.get('eps')
        seed = kwargs.get('seed')
        profile_mode = {'direct': 'k', 'recursive': 'r'}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"
        context = kahypar.Context()
        context.loadINIconfiguration(join(KAHYPAR_PROFILE_DIR, profile))
        context.setK(K)
        context.setSeed(seed)
        context.setEpsilon(kwargs.get('epsilon', eps * (K - 1)))
        context.suppressOutput(kwargs.get('quiet', True))
        return context

    def _sub(self, graph, nodes, sub_nodes, new_name):
        inds = [False] + [n in sub_nodes for n in nodes]
        out_nodes = [n for n in nodes if n not in sub_nodes] + [new_name]
        subb = graph[inds]
        rest = graph[numpy.invert(inds)]
        open_edges = numpy.any(subb, axis=0) & numpy.any(rest, axis=0)
        return numpy.append(open_edges[None, :], subb,
                            axis=0), numpy.append(rest,
                                                  open_edges[None, :],
                                                  axis=0), out_nodes

    def _init_hypergraph_tanner(self, graph, k=2):
        nodes = list(range(1, graph.shape[0]))
        edges = list(range(graph.shape[1]))
        hyperedge_indices = [0]
        hyperedges = []
        edge_weights = []
        for edge in range(graph.shape[1]):
            hyperedges += list(numpy.where(graph[1:, edge])[0])
            hyperedge_indices.append(len(hyperedges))
            edge_weights.append(1)
        hp = kahypar.Hypergraph(len(nodes), len(edges), hyperedge_indices,
                                hyperedges, k, edge_weights, []), nodes, edges
        return hp

    def _refresh(self, graph):
        return graph[:, numpy.sum(graph, axis=0) > 0]
