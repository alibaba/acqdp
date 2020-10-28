import networkx as nx


class UndirectedContractionTree:
    """Contraction tree corresponding to a pairwise sequential contraction order, with interfaces for branch flipping
    and merging.

    TODO: Merge this class with :class:`ContractionTree` for a unified interface on contraction trees.
    """

    def __init__(self, eq, path):
        lhs, rhs = eq.replace(' ', '').split('->')
        lhs = lhs.split(',')
        self.n = len(lhs)
        self.open_subscripts = set(rhs)
        if self.n == 1:
            raise ValueError('Cannot construct undirected contraction tree with only one operand')
        self.graph = nx.Graph()
        operands = list(range(self.n))
        self.graph.add_nodes_from((i, {'subscripts': set(x)}) for i, x in enumerate(lhs))
        t = self.n
        for i, j in path[:-1]:
            if i > j:
                i, j = j, i
            v = operands.pop(j)
            u = operands.pop(i)
            self.graph.add_edge(u, t)
            self.graph.add_edge(v, t)
            self.graph.nodes[u]['parent'] = self.graph.nodes[v]['parent'] = t
            operands.append(t)
            t += 1
        u, v = self.root = tuple(operands)  # There should be two operands left
        self.graph.add_edge(u, v)
        self.graph.nodes[u]['parent'] = v
        self.graph.nodes[v]['parent'] = u
        for u, v in self.graph.edges:
            self.preprocess_edge(u, v)
            self.preprocess_edge(v, u)
        self.cost = self.root_cost = 0
        self.compute_root_cost()
        for v in range(self.n, self.n * 2 - 2):
            self.compute_node_cost(v)
        self.detect_stem()

    def is_leaf(self, v):
        return v < self.n

    def preprocess_edge(self, u, v):
        if v not in self.graph[u][v]:
            if self.is_leaf(v):
                self.graph[u][v][v] = {'subscripts': self.graph.nodes[v]['subscripts']}
            else:
                children = [c for c in self.graph[v] if c != u]
                assert len(children) == 2
                d0, d1 = (self.preprocess_edge(v, c) for c in children)
                self.graph[u][v][v] = {'subscripts': d0['subscripts'] | d1['subscripts']}
        return self.graph[u][v][v]

    def open_subscripts_at_edge(self, u, v):
        edge = self.graph[u][v]
        if self.is_leaf(v):
            return edge[v]['subscripts']
        return edge[v]['subscripts'] & (edge[u]['subscripts'] | self.open_subscripts)

    def compute_cost(self, s0, s1, s):
        res = 2 ** len(s0 | s1 | s)
        if (s0 | s1) - s:
            res *= 2
        return res

    def compute_root_cost(self):
        u, v = self.root
        s0 = self.open_subscripts_at_edge(u, v)
        s1 = self.open_subscripts_at_edge(v, u)
        self.cost -= self.root_cost
        self.root_cost = self.compute_cost(s0, s1, self.open_subscripts)
        self.cost += self.root_cost

    def compute_node_cost(self, v):
        parent = self.graph.nodes[v]['parent']
        children = [c for c in self.graph[v] if c != parent]
        assert len(children) == 2
        s0, s1 = (self.open_subscripts_at_edge(v, c) for c in children)
        if 'cost' in self.graph.nodes[v]:
            self.cost -= self.graph.nodes[v]['cost']
        self.graph.nodes[v]['cost'] = self.compute_cost(s0, s1, self.open_subscripts_at_edge(parent, v))
        self.cost += self.graph.nodes[v]['cost']

    def get_internal_path(self, v):
        if self.is_leaf(v):
            return []
        parent = self.graph.nodes[v]['parent']
        c0, c1 = [c for c in self.graph[v] if c != parent]
        return self.get_internal_path(c0) + self.get_internal_path(c1) + [(c0, c1, v)]

    def get_path(self):
        operands = list(range(self.n))
        res = []
        for r in self.root:  # Two halves of the contraction tree
            for c0, c1, v in self.get_internal_path(r):
                res.append((operands.index(c0), operands.index(c1)))
                operands.remove(c0)
                operands.remove(c1)
                operands.append(v)
        return res + [(0, 1)]

    def switch_edges(self, e0, e1):
        """
        Switch the edges e0 = (u, c0) and e1 = (v, c1) by connecting
        c0 to v and c1 to u instead. The path c0 -- u -- v -- c1 must
        be a path in the tree.
        This might break the stem, so if one or both of e0 and e1 may
        be on the stem, please use `switch_branches`, `merge_branches`,
        `unmerge_branches`, or fix the stem manually.
        """
        for (u, c0), (v, c1) in (e0, e1), (e1, e0):
            self.graph.add_edge(v, c0)
            self.graph[v][c0][c0] = self.graph[u][c0][c0]
            self.graph[v][c0][v] = self.graph[u][c0][u]
            self.graph.remove_edge(u, c0)
            if self.graph.nodes[c0]['parent'] == u:
                self.graph.nodes[c0]['parent'] = v
            if self.graph.nodes[u]['parent'] == c0:
                self.graph.nodes[v]['parent'] = c0
                self.graph.nodes[u]['parent'] = v
            if set(self.root) == {u, c0}:
                self.root = (v, c0)
        self.graph[u][v].clear()
        self.preprocess_edge(u, v)
        self.preprocess_edge(v, u)
        self.compute_node_cost(u)
        self.compute_node_cost(v)
        if set(self.root) == {u, v}:
            self.compute_root_cost()

    def step_root(self, u, v):
        if u not in self.root:
            raise ValueError('The new root must be adjacent to the old root at vertex u')
        self.root = u, v
        self.graph.nodes[u]['parent'] = v
        self.compute_node_cost(u)
        self.compute_root_cost()

    def detect_stem_in_subtree(self, u, v):
        """Detect the heaviest path in the subtree descending from the edge (u, v), *and* the heaviest such path that
        ends at v.

        Return format is (cost, endpoint, endpoint), (cost, endpoint).
        """
        if self.is_leaf(v):
            return (0, v, v), (0, v)
        c0, c1 = [c for c in self.graph[v] if c != u]
        cost_v = self.graph.nodes[v]['cost']
        res0, (cost0, end0) = self.detect_stem_in_subtree(v, c0)
        res1, (cost1, end1) = self.detect_stem_in_subtree(v, c1)
        cost, end = max((cost0, end0), (cost1, end1))
        res = max(res0, res1, (cost0 + cost1 + cost_v, end0, end1), key=lambda x: x[0])
        return res, (cost + cost_v, end)

    def detect_stem(self):
        # Just choose any leaf as the root to make things easier.
        (cost, end0, end1), _ = self.detect_stem_in_subtree(0, self.graph.nodes[0]['parent'])
        # Maybe not the most efficient, but convenient enough
        self.stem = nx.shortest_path(self.graph, end0, end1)
        return self.stem

    def switch_branches(self, i):
        """Switch the branches at nodes i and i+1 of the stem."""
        t0, u, v, t1 = self.stem[i - 1:i + 3]
        c0, = set(self.graph[u]) - {t0, v}
        c1, = set(self.graph[v]) - {u, t1}
        self.switch_edges((u, c0), (v, c1))

    def merge_branches(self, i):
        """Merge the branches at nodes i and i+1 of the stem."""
        t0, u, v, t1 = self.stem[i - 1:i + 3]
        c0, = set(self.graph[u]) - {t0, v}
        self.switch_edges((u, c0), (v, t1))
        self.stem.pop(i + 1)
        return c0

    def unmerge_branches(self, i, c0=None):
        """Split the branch at node i of the stem into two branches, with the sub-branch c0 now directly connecting to
        node i of the stem and the other sub-branch connecting to node i+1 of the stem.

        By default, use the sub-branch with smaller id as c0.
        """
        t0, u, t1 = self.stem[i - 1:i + 2]
        v, = set(self.graph[u]) - {t0, t1}
        if c0 is None:
            c0 = min(c for c in self.graph[v] if c != u)
        self.switch_edges((u, t1), (v, c0))
        self.stem.insert(i + 1, v)
