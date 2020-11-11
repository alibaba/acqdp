import numpy as np
import opt_einsum
from acqdp.tensor_network import ContractionTask, defaultOrderResolver


class Compiler:
    """Compile a :class:`ContractionScheme` indicating contraction scheme and sliced edges into a
    :class:`ContractionTask`, a hardware-aware format that can be readily used for tensor network contraction.

    Typically, a :class:`ContractionScheme` found by one of the :class:`OrderFinder` only aims to minimize the theoretical
    floating number operations and / or largest size of intermediate tensors. Although those are typically accurate indicators
    of the overall contracion cost, they fail to accomodate inefficiencies from elsewhere. The compiler takes care of
    machine-related inefficiencies by slightly modifying the order.

    :ivar do_patch: If set to false, the patching and reorganization of the contraction order below will be skipped.
    :ivar patch_size: Whenever two adjacent branches both have size less than the patch size, the branches are merged
        together. This is to avoid inefficiency called by skewed tensor shapes in tensor multiplication, with a slight
        sacrifice in floating number operations.
    :ivar memory: Memory constraint for the machine(s) running the contraction, in GigaBytes. Set to 16GiB by default.
    :ivar reorg_thres: Threshold for contraction order reorganization. The order is seperated into small tensor
        multiplications and large tensor multiplications. All pairwise tensor multiplications involving tensors with size
        less than `reorg_thres` will be put forward whenever possible.
    """

    def __init__(self,
                 reorg_thres=23,
                 patch_size=5,
                 do_patch=False,
                 memory=16,
                 **kwargs):
        self.reorg_thres = reorg_thres
        self.patch_size = patch_size
        self.do_patch = do_patch
        self.memory = memory

    def compile(self, tn, scheme, **kwargs):
        """
        :param tn: The :class:`TensorNetwork` the input contraction scheme is based on.
        :type tn: :class:`TensorNetwork`
        :param scheme: The contraction scheme from which the task is generated.
        :type scheme: :class:`ContractionScheme`

        :returns: :class:`ContractionTask` -- The compiled contraction task ready for execution.
        """
        tn = tn._expand_and_delta()
        cost = scheme.cost
        if cost.t * 3 * 2**4 > self.memory * 2**30:
            raise RuntimeError("The contraction cannot be done on the current machine."
                               + " Try finding a more space-efficient order, or slice the order to fit in the memory.")
        res = self._generate_template(tn, scheme=scheme, **kwargs)
        res.set_data({
            node: tn.network.nodes[(0, node)]['tensor'].contract()
            for node in tn.nodes_by_name
        })
        res.update_fix(tn)
        return res

    def _generate_template(self, tn, scheme, test=0, **kwargs):
        inputs = {node[1]: [None] for node in tn.nodes}
        data_ref = {(0, node): inputs[node] for node in inputs}
        order, split = scheme.order, scheme.slices
        split = [(1, i) for i in split]
        tn_copy = tn.copy().expand(recursive=True)
        tn_copy.open_edges += [i[1] for i in split]
        tn_copy.fix()
        for edge_name in list(tn_copy.edges_by_name):
            if (len(tn_copy.network[(1, edge_name)])
                    == 0) and edge_name not in tn_copy.open_edges:
                tn_copy.network.remove_node((1, edge_name))
        init_order, final_order = self._order_patch(tn_copy.copy(), order, split,
                                                    **kwargs)
        k = self._generate_template_fix(tn, [], data_ref)
        if isinstance(k, ContractionTask):
            return k
        lst, fix_dict = k

        for item in init_order:
            if item[1] == '#':
                break
            stn_name, stn = tn_copy.encapsulate(item[0], stn_name=item[1])
            lst += self._generate_template_basic(stn, data_ref, (0, stn_name))

        cnt = len(lst)

        fix_lst, fix_dict_new = self._generate_template_fix(
            tn_copy, split, data_ref)
        lst += fix_lst
        fix_dict.update(fix_dict_new)

        tn_copy.open_edges = [
            i for i in tn_copy.open_edges if (1, i) not in split
        ]
        for i in split:
            tn_copy.fix_edge(i[1])
        tn_copy.fix()

        lst += self._generate_template_basic(tn_copy, data_ref, '#')
        _, _, shapes = tn_copy.subscripts()
        if len(final_order) > 0:
            path = []
            lst_nodes = list(n[1] for n in tn_copy.nodes)
            for o in final_order:
                path.append(
                    (lst_nodes.index(o[0][0]), lst_nodes.index(o[0][1])))
                lst_nodes.remove(o[0][0])
                lst_nodes.remove(o[0][1])
                lst_nodes.append(o[1])
            expr = opt_einsum.contract_expression(lst[-1][3]['subscripts'],
                                                  *shapes,
                                                  optimize=path)
            lst[-1][3]['expr'] = expr
        res = ContractionTask(commands=lst,
                              fix_dict=fix_dict,
                              inputs=inputs,
                              output=data_ref['#'],
                              shape=tn.shape,
                              open_edges=tn.open_edges,
                              sub_outputs=tn_copy.open_edges,
                              cnt=cnt)
        return res

    def _generate_template_fix(self, tn, split, data_ref):
        lst = []

        fix_dict = self._generate_template_edge_fix(tn, split)
        if not isinstance(fix_dict, dict):
            return fix_dict
        for node in tn.nodes:
            fix_idx = []
            fix_to = []
            for i, edge in enumerate(tn.network.nodes[node]['edges']):
                if (1, edge) in fix_dict:
                    fix_idx.append(i)
                    fix_to.append(fix_dict[(1, edge)][1])
            if len(fix_idx) > 0:
                new_res = [None]
                lst.append(('f', data_ref[node], new_res, {'fix_idx': fix_idx, 'fix_to': fix_to}))
                data_ref[node] = new_res
        res_dict = {(k, tuple(fix_dict[k][0])): fix_dict[k][1] for k in fix_dict}
        return lst, res_dict

    def _generate_template_edge_fix(self, tn, split):
        fix_dict = {}
        for edge in tn.edges:
            if edge in split:
                tn.network.nodes[edge]['fix_to'] = list(range(tn.network.nodes[edge]['dim']))
                fix_dict[edge] = (list(range(tn.network.nodes[edge]['dim'])), [0])
            elif 'fix_to' in tn.network.nodes[edge]:
                ll = len(tn.network.nodes[edge]['fix_to'])
                if ll == 0:
                    return ContractionTask(output=[(0, np.zeros(tn.shape))])
                else:
                    fix_dict[edge] = (list(tn.network.nodes[edge]['fix_to']), list(tn.network.nodes[edge]['fix_to']))
        return fix_dict

    def _generate_template_delta(self, tn, data_ref):
        open_edge_names = {}
        edges = tn.edges_by_name_from_nodes_by_name()
        if len(tn.nodes) == 0:
            id_node = np.array(1.)
            tn.add_node('*', [], id_node)
            data_ref[(0, '*')] = [(0, id_node)]
        for (i, edge) in enumerate(tn.open_edges):
            if edge not in open_edge_names:
                open_edge_names.update({edge: 0})
            else:
                open_edge_names[edge] += 1
                new_node = np.eye(tn.network.nodes[(1, edge)]['dim'])
                data_ref[(0, ("*", edge, open_edge_names[edge]))] = [(0, new_node)]
                tn.add_node(("*", edge, open_edge_names[edge]), [edge, (edge, open_edge_names[edge])], new_node)
                tn.open_edges[i] = (edge, open_edge_names[edge])
        for edge in open_edge_names:
            diag = None
            if edge not in edges and open_edge_names[edge] == 0:
                diag = [1.] * tn.network.nodes[(1, edge)]['dim']
            if 'fix_to' in tn.network.nodes[(1, edge)]:
                diag = [1. if i in tn.network.nodes[(1, edge)]['fix_to']
                        else 0 for i in range(tn.network.nodes[(1, edge)]['dim'])]
                tn.network.nodes[(1, edge)].pop('fix_to')
            if diag is not None:
                new_node = np.array(diag)
                data_ref[(0, ('*', edge, 0))] = [(0, new_node)]
                tn.add_node(('*', edge, 0), [edge], new_node)

    def _generate_template_basic(self, tn, data_ref, new_name, **kwargs):
        new_res = [None]
        data_ref[new_name] = new_res
        if new_name == '#':
            self._generate_template_delta(tn, data_ref)
        lhs = [data_ref[node] for node in tn.nodes]

        return [('c', lhs, new_res, self._generate_template_einsum(
            tn, subscripts=tn.subscripts(), **kwargs), [node for node in tn.nodes])]

    def _generate_template_einsum(self, tn, subscripts, **kwargs):
        lhs_str, rhs_str, _ = subscripts
        command = ','.join(lhs_str) + '->' + rhs_str
        return {'subscripts': command, 'dtype': tn.dtype}

    def _order_patch(self, tn, order, split_edges, **kwargs):
        if self.do_patch:
            from acqdp.tensor_network.contraction_tree import ContractionTree
            tn_copy = tn.copy()
            tn_copy.open_edges += [i[1] for i in split_edges]
            tn_copy.fix()
            for edge_name in list(tn_copy.edges_by_name):
                if (len(tn_copy.network[(1, edge_name)])
                        == 0) and edge_name not in tn_copy.open_edges:
                    tn_copy.network.remove_node((1, edge_name))
            assert tn_copy.dtype == tn.dtype
            order = ContractionTree.from_order(tn_copy, order).full_order
            init_order, final_order = self._reorg_order(tn_copy, order)
            tn_copy.open_edges = [
                i for i in tn_copy.open_edges if (1, i) not in split_edges
            ]
            for i in split_edges:
                tn_copy.fix_edge(i[1])
            tn_copy.fix()
            if len(final_order) > 0:
                final_order = self._patch_order(tn_copy, final_order)
            return init_order, final_order
        else:
            return [], order

    def _reorg_order(self, graph, order):
        graph.fix()
        size_dic = {}
        subs, order = defaultOrderResolver.order_to_subscripts_tree(
            graph, order)
        assert len(subs) == len(order), f'{len(subs)}, {len(order)}'
        for i, o in enumerate(order):
            size_dic[o[1]] = len(subs[i][1])
            for j, k in enumerate(o[0]):
                size_dic[k] = len(subs[i][0][j])
        init_order = []
        node_set = set([a[1] for a in graph.nodes])
        while True:
            for o in order:
                if size_dic[o[1]] < self.reorg_thres and np.all(
                        [a in node_set for a in o[0]]):
                    init_order.append(o)
                    order.remove(o)
                    for a in o[0]:
                        node_set.remove(a)
                    node_set.add(o[1])
                    graph.encapsulate(nodes=o[0], stn_name=o[1])
                    break
            else:
                break
        return init_order, order

    def _patch_order(self, graph, order):
        from acqdp.tensor_network.contraction_tree import ContractionTree
        graph_copy = graph.copy()
        graph_copy.fix()
        tree = ContractionTree.from_order(graph_copy, order)

        res = tree.patch(self.patch_size)
        return res
