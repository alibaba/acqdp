from acqdp.tensor_network.contraction import ContractionCost
import opt_einsum
import sys
import networkx
from networkx.algorithms import max_weight_matching

sys.setrecursionlimit(10**4)

CONTRACTION_TREE_PARAMS = {
    'stem_size': 23,
    'connection': 4,
    'k': 14
}


class ContractionTree:
    """Contraction tree corresponding to a pairwise sequential contraction, with interfaces for branch flipping and
    merging.

    TODO: Merge with :class:`UndirectedContractionTree` for a unified interface for contraction trees.
    """
    @classmethod
    def from_order(cls, tn, order):
        tn_copy = tn.copy()
        tree_dic = {}
        edge_list = list(e[1] for e in tn_copy.edges)
        for node in tn_copy.nodes:
            tree_dic[node[1]] = cls(subscripts=''.join([opt_einsum.get_symbol(edge_list.index(e))
                                                        for e in tn_copy.network.nodes[node]['edges']]), name=node[1])
        for o in order:
            stn_name, _ = tn_copy.encapsulate(o[0], stn_name=o[1])
            tree_dic[stn_name] = cls(subscripts=''.join([opt_einsum.get_symbol(edge_list.index(
                e)) for e in tn_copy.network.nodes[(0, stn_name)]['edges']]), name=o[1], children=[tree_dic[i] for i in o[0]])
        return tree_dic['#']

    def __init__(self, subscripts=None, children=None, name=None):
        self.name = name
        self.subs = subscripts
        self.children = children
        if self.children is not None:
            self.all_subscripts = set()
            for c in self.children:
                # c.parent = self
                self.all_subscripts |= set(c.subs)
            self.children = sorted(self.children, key=lambda x: len(x.subs))
        else:
            self.all_subscripts = self.subs

    @property
    def subscripts(self):
        if self.children is None:
            return []
        res = [[[c.subs for c in self.children], self.subs]]
        return res

    @property
    def full_subscripts(self):
        res = []
        if self.children is None:
            return res
        for c in self.children[::-1]:
            res += c.full_subscripts
        return res + self.subscripts

    @property
    def order(self):
        if self.children is None:
            return []
        return [[[c.name for c in self.children], self.name]]

    @property
    def full_order(self):
        res = []
        if self.children is None:
            return res
        assert len(self.children) <= 2 and self not in self.children
        for c in self.children[::-1]:
            res += c.full_order
        return res + self.order

    @property
    def step_cost(self):
        if self.children is None:
            return ContractionCost(1, 1)
        else:
            return ContractionCost(2**len(self.all_subscripts), 2**len(self.subs))

    @property
    def total_cost(self):
        res = self.step_cost
        if self.children is not None:
            for c in self.children:
                res += c.total_cost
        return res

    def flip_branch(self):
        flg = False
        if len(self.subs) >= CONTRACTION_TREE_PARAMS['stem_size'] and\
           self.children is not None and\
           self.children[1].children is not None:
            stem0 = self.children[1].children[1]
            stem1 = self.children[1]
            branch0 = self.children[1].children[0]
            branch1 = self.children[0]
            imag_stem1_remain = ''.join(set(stem0.subs).union(branch1.subs).intersection(set(self.subs).union(branch0.subs)))
            if len(imag_stem1_remain) < len(stem1.subs) or (
                    len(imag_stem1_remain) == len(stem1.subs) and branch0.subs > branch1.subs):
                flg = True
                imag_stem1 = ContractionTree(children=[branch1, stem0], subscripts=imag_stem1_remain, name=stem1.name)
                self.children = [branch0, imag_stem1]
                self.all_subscripts = set(imag_stem1.subs).union(branch0.subs)
        if self.children is not None:
            for c in self.children:
                flg |= c.flip_branch()
        return flg

    def branch_info(self, c=True):
        res = []
        if c is True:
            if self.children is None:
                return res
            if max(len(self.children[0].subs), len(self.children[1].subs)) >= CONTRACTION_TREE_PARAMS['stem_size']:
                res = [[self, self.subs]]
                c = False
                for cc in self.children:
                    res.append([[cc, cc.subs]] + cc.branch_info(c))
            else:
                return self.children[0].branch_info() + self.children[1].branch_info()

        if len(self.subs) >= CONTRACTION_TREE_PARAMS['stem_size']:
            if self.children is not None:
                m = set(self.children[0].subs).difference(self.children[1].subs)
                n = set(self.children[0].subs).intersection(self.children[1].subs)
                res.append([self.children[0], ''.join(n), ''.join(m), self.children[1]])
            if self.children is not None:
                for cc in self.children:
                    res += cc.branch_info(c)
        return res

    def branches_to_digraph(self, branches):
        k = branches[0]
        if len(branches) <= 3:
            return None
        g = networkx.MultiDiGraph()
        g.add_node(0, branch=k[0])
        edges_dic = {a: 0 for a in k[1]}
        self.inter_names = []
        ll = len(branches) - 1
        for i, ss in enumerate(branches):
            if len(ss) > 2:
                self.inter_names.append(ss[3].name)
                g.add_node(i + 1, branch=ss[0])
                for n in ss[2]:
                    g.add_edge(i + 1 if i < ll - 1 else -1, edges_dic[n], name=n)
                    del edges_dic[n]
                for n in ss[1]:
                    edges_dic[n] = i + 1
        g.nodes[-1]['branch'] = ss[3]
        for n in edges_dic:
            g.add_edge(-1, edges_dic[n], name=n)
        return g

    def reconstruct_order(self, g):
        curr_node = g.nodes[-1]['branch']
        for node in [i for i in list(g.nodes) if i != -1][:0:-1]:
            nn = g.nodes[node]['branch']
            subs = set(nn.subs).difference(curr_node.subs) | set(curr_node.subs).difference(nn.subs)
            new_node = ContractionTree(children=[nn, curr_node], subscripts=subs, name=self.inter_names[0])
            del self.inter_names[0]
            curr_node = new_node
        g.nodes[0]['branch'].children = curr_node.children

    def merge_nodes(self, width):
        br = self.branch_info()
        if len(br[1]) >= len(br[2]):
            brb = [br[2], br[1]]
        else:
            brb = [br[1], br[2]]
        m0, m1 = set(brb[0][0][1]), set(brb[1][0][1])
        i = 0
        while (len(m0) + len(m1) - 2 * len(m0.intersection(m1)) <= 12) and i < len(brb):
            i += 1
            m0 = m0.union(brb[0][i][1]).difference(brb[0][i][2])
        if i > 1:
            k1 = brb[0][i][0]
            p1 = brb[1][0][0]
            br[0][0].children = [k1, brb[0][0][0]]
            brb[0][i - 1][3].children = [brb[0][i][3], p1]
            brb[0][i] = [brb[0][i][3], ''.join(m0)]

            brb[0] = brb[0][i:]
        # assert False
        for brk in brb:
            g = self.branches_to_digraph(brk)
            if g is None:
                continue
            while True:
                a = networkx.Graph()
                g_list = list(g.nodes)
                for i, u in enumerate(g_list):
                    if u == 0 or u == -1 or i >= len(g_list) - 2:
                        continue
                    vs = [list(g_list)[i + 1] if list(g_list)[i + 1] != -1 else list(g_list)[-2]]
                    vv = sorted([i for i in list(g.predecessors(u)) if i != -1])
                    if len(vv) > 0 and vv[0] != vs[0]:
                        vs.append(vv[0])
                        # if v == -1: continue
                    for v in vs:
                        if g.in_degree(u) <= width and g.out_degree(v) <= width and g.out_degree(u) <= g.in_degree(u):
                            ustr = g.nodes[u]['branch'].subs
                            vstr = g.nodes[v]['branch'].subs
                            kstr = set(ustr).symmetric_difference(vstr)
                            if 2**len(ustr) + 2**len(vstr) >= 2**(len(kstr)) / CONTRACTION_TREE_PARAMS['k']:
                                a.add_edge(u, v, weight=2**(-len(kstr)))

                matching = max_weight_matching(a)
                if len(matching) == 0:
                    break
                for u, v in matching:
                    u, v = max(u, v), min(u, v)
                    bu = g.nodes[u]['branch']
                    bv = g.nodes[v]['branch']
                    subs = ''.join(set(bu.subs).difference(bv.subs) | set(bv.subs).difference(bu.subs))
                    for node in g.successors(v):
                        if node != u:
                            for e in g[v][node]:
                                g.add_edge(u, node, name=g[v][node][e]['name'])
                    for node in g.predecessors(v):
                        if node != u:
                            for e in g[node][v]:
                                g.add_edge(node, u, name=g[node][v][e]['name'])
                    name = self.inter_names[0]
                    del self.inter_names[0]
                    g.nodes[u]['branch'] = ContractionTree(children=[bu, bv], name=name, subscripts=subs)
                    g.remove_node(v)

                a = networkx.Graph()
                g_list = list(g.nodes)
                for i, u in enumerate(g_list):
                    if u == 0 or u == -1 or i <= 1:
                        continue
                    vs = [g_list[i - 1]]
                    vv = sorted([i for i in list(g.successors(u)) if i != 0])
                    if len(vv) > 0 and vv[-1] != vs[0]:
                        vs.append(vv[-1])
                    for v in vs:
                        # if v == 0: continue
                        if g.out_degree(u) <= width and g.in_degree(v) <= width and g.in_degree(u) <= g.out_degree(u):
                            ustr = g.nodes[u]['branch'].subs
                            vstr = g.nodes[v]['branch'].subs
                            kstr = set(ustr).symmetric_difference(vstr)
                            if 2**len(ustr) + 2**len(vstr) >= 2**(len(kstr)) / CONTRACTION_TREE_PARAMS['k']:
                                a.add_edge(u, v, weight=2**(-len(kstr)))

                matching = max_weight_matching(a)
                if len(matching) == 0:
                    break
                for u, v in matching:
                    u, v = min(u, v), max(u, v)
                    bu = g.nodes[u]['branch']
                    bv = g.nodes[v]['branch']
                    subs = ''.join(set(bu.subs).difference(bv.subs) | set(bv.subs).difference(bu.subs))
                    for node in g.successors(v):
                        if node != u:
                            for e in g[v][node]:
                                g.add_edge(u, node, name=g[v][node][e]['name'])
                    for node in g.predecessors(v):
                        if node != u:
                            for e in g[node][v]:
                                g.add_edge(node, u, name=g[node][v][e]['name'])
                    name = self.inter_names[0]
                    del self.inter_names[0]
                    g.nodes[u]['branch'] = ContractionTree(children=[bu, bv], name=name, subscripts=subs)
                    g.remove_node(v)
            self.reconstruct_order(g)

    def patch(self, width=5):
        self.merge_nodes(width)
        return self.full_order
