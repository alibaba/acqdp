from .circuit import (Operation,
                      ImmutableOperation,
                      PureOperation,
                      Circuit,
                      ControlledOperation,
                      Controlled,
                      ControlledCircuit,
                      CompState,
                      CompMeas,
                      SuperPosition)

from acqdp.tensor_network.tensor_valued import TensorValued

from acqdp.tensor_network.tensor_sum import TensorSum

from acqdp.tensor_network.tensor_network import TensorNetwork


class Converter:
    """Generic converter class, with method mapping a circuit to a tensor network."""
    @classmethod
    def convert_pure(cls, operation: Operation) -> TensorValued:
        """Do a pure conversion for noiseless circuit described as a :class:`Operation` object to a tensor network as a
        :class:`TensorValued` object.

        :param operation: the noisy circuit to convert.
        :type operation: :class:`Operation`.
        :returns:  :class:`TensorValued` -- the tensor network to describe the input noisy circuit.
        """
        if not operation.is_pure:
            raise ValueError("Invalid operation: trying to do pure conversion for noisy circuit")
        if isinstance(operation, ControlledCircuit):
            op = operation.circuit
            c = Circuit()
            qubits = op.all_qubits
            for gate in op.operations_by_name:
                c.append(Controlled(op.operations_by_name[gate]["operation"],
                                    conditioned_on=operation.conditioned_on),
                         [0] + [1 + qubits.index(i) for i in op.operations_by_name[gate]['qubits']])
            return c.tensor_pure
        elif isinstance(operation, ControlledOperation):
            tn = TensorNetwork()
            tn.add_node("TC", tensor=operation._tensor_control)
            out_edges = []
            in_edges = []
            lst_ctrl, len_ctrl = operation._indices_with_property("^c$")
            lst_out, len_out = operation._indices_with_property(".*o$")
            lst_in, len_in = operation._indices_with_property("^i")
            for i in operation._output_indices[0]:
                if i in lst_ctrl:
                    out_edges.append(lst_ctrl.index(i))
                else:
                    out_edges.append(len_ctrl + lst_out.index(i))
            for i in operation._input_indices[0]:
                if i in lst_ctrl:
                    in_edges.append(lst_ctrl.index(i))
                else:
                    in_edges.append(len_ctrl + len_out + lst_in.index(i))
            tn.open_edges = out_edges + in_edges
            return tn
        elif isinstance(operation, PureOperation):
            return operation._tensor_pure
        elif isinstance(operation, SuperPosition):
            ts = TensorSum()
            for i in range(len(operation.coefs)):
                ts.add_term(i, operation.coefs[i] * operation.operations[i].tensor_pure)
            return ts
        else:
            if not isinstance(operation, Circuit):
                raise ValueError("Invalid operation: operation is not yet supported")
            # This cannot happen normally because Circuit.is_pure checks
            # Circuit.is_valid, but is left in just in case someone subclasses
            # Circuit and overrides is_pure.
            if not operation.is_valid:  # pragma: no cover
                raise ValueError("Invalid operation: trying to convert an invalid circuit into a tensor network")
            dic = {}
            dic_in = {}
            dic_out = {}
            tn = TensorNetwork()
            all_edges = set()
            for time_step in list(operation.operations_by_time)[::-1]:
                for idx, gate in enumerate(operation.operations_by_time[time_step]):
                    op = operation.operations_by_name[gate]['operation']
                    qubits = operation.operations_by_name[gate]['qubits']
                    out_edges = []
                    in_edges = []
                    for i in op._output_indices[0]:
                        q = qubits[i]
                        if q not in dic:
                            dic_out[q] = (time_step + 1, idx, q)
                            out_edges.append(dic_out[q])
                        else:
                            edge = dic.pop(q)
                            out_edges.append(edge)
                            dic_in.pop(q)
                    for i in op._input_indices[0]:
                        q = qubits[i]
                        dic[q] = (time_step, idx, q)
                        in_edges.append(dic[q])
                        dic_in[q] = dic[q]
                    for edge in out_edges + in_edges:
                        if edge not in all_edges:
                            tn.add_edge(edge, bond_dim=2)
                            all_edges.add(edge)
                    if op in CompState:
                        tn.fix_edge(out_edges[0], CompState.index(op))
                    elif op in CompMeas:
                        tn.fix_edge(in_edges[0], CompMeas.index(op))
                    else:
                        tn.add_node(gate, out_edges + in_edges, op.tensor_pure)
            tn.open_edges = [dic_out[i] for i in sorted(dic_out)] +\
                            [dic_in[i] for i in sorted(dic_in)]
            return tn

    @classmethod
    def convert_control(cls, operation: PureOperation) -> TensorValued:
        """Do a control conversion for noisy circuit described as a :class:`Operation` object to a tensor network as a
        :class:`TensorValued` object.

        :param operation: the noisy circuit to convert.
        :type operation: :class:`Operation`.
        :returns:  :class:`TensorValued` -- the tensor network to describe the input noisy circuit.
        """
        if not operation.is_pure:
            raise ValueError("Noisy operation does not have pure block-diagonal form")
        if isinstance(operation, ControlledOperation):
            return operation._tensor_control
        else:
            return operation.tensor_pure

    @classmethod
    def convert_density(cls, operation: Operation) -> TensorValued:
        """Do a density conversion for noisy circuit described as a :class:`Operation` object to a tensor network as a
        :class:`TensorValued` object.

        :param operation: the noisy circuit to convert.
        :type operation: :class:`Operation`.
        :returns:  :class:`TensorValued` -- the tensor network to describe the input noisy circuit.
        """
        if isinstance(operation, PureOperation):
            tp = operation.tensor_pure
            tn = TensorNetwork()
            len_tensor = len(tp.shape)
            tn.add_node("TP", list(range(len_tensor)), tp)
            tn.add_node("~TP", list(range(len_tensor, 2 * len_tensor)), (~tp).expand())
            tn.open_edges = list(range(2 * len_tensor))
            return tn
        elif isinstance(operation, ImmutableOperation):
            return operation._tensor_density
        else:
            if not isinstance(operation, Circuit):
                raise ValueError("Invalid operation: operation is not yet supported")
            if not operation.is_valid:
                raise ValueError('Invalid operation')
            dic = {}
            tn = TensorNetwork()
            dic_out = {}
            dic_in = {}
            for time_step in list(operation.operations_by_time):
                for idx, gate in enumerate(operation.operations_by_time[time_step]):
                    op = operation.operations_by_name[gate]["operation"]
                    qubits = operation.operations_by_name[gate]['qubits']
                    out_edges_p = []
                    in_edges_p = []
                    out_edges_d = []
                    in_edges_d = []

                    for i in op._input_indices[0]:
                        q = qubits[i]
                        if q not in dic:
                            in_edges_p.append((time_step, idx, q, 'p'))
                            in_edges_d.append((time_step, idx, q, 'd'))
                            dic_in[q] = [time_step, idx, q]
                        else:
                            edge = list(dic.pop(q))
                            in_edges_p.append(tuple(edge + ['p']))
                            in_edges_d.append(tuple(edge + ['d']))
                            dic_out.pop(q)
                    for i in op._output_indices[0]:
                        q = qubits[i]
                        dic[q] = (time_step + 1, idx, q)
                        out_edges_p.append((time_step + 1, idx, q, 'p'))
                        out_edges_d.append((time_step + 1, idx, q, 'd'))
                        dic_out[q] = [time_step + 1, idx, q]
                    tn.add_node(gate, out_edges_p + in_edges_p + out_edges_d + in_edges_d, op.tensor_density)
            tn.open_edges = [tuple(dic_out[i] + ['p']) for i in sorted(dic_out)] +\
                            [tuple(dic_in[i] + ['p']) for i in sorted(dic_in)] +\
                            [tuple(dic_out[i] + ['d']) for i in sorted(dic_out)] +\
                            [tuple(dic_in[i] + ['d']) for i in sorted(dic_in)]
            return tn
