import numpy as np
import collections
import re
from copy import copy
import math
# from .myabc import MN
from acqdp.tensor_network.tensor_valued import TensorValued
from acqdp.tensor_network.tensor import Tensor
from acqdp.tensor_network.tensor_network import TensorNetwork
from typing import List, Optional


INDENT = "  "
"""The unit of indentation used for :meth:`Operation.tree_string`."""


class Operation(object):
    """Base class for quantum opertations.

    By itself, it can represent a generic quantum operation, although one would
    not be able to do much with it. Usually, one should use
    :class:`ImmutableOperation` (or more commonly, one of its subclasses) for
    simple quantum operations with an explicit tensor representation, and
    :class:`Circuit` for quantum operations better represented as a composition
    of simpler operations. (Technically an :class:`ImmutableOperation` can
    represent a complex operation by using a :class:`TensorNetwork` as the data,
    but that would be an uncommon use case.)

    :ivar name: Name of the quantum operation. Defaults to "GenOp".
    :vartype name: str, optional
    """

    def __init__(self, name="GenOp"):
        self.name = name

    def __str__(self) -> str:  # pragma: no cover
        return str(self.name)

    def tree_string(self, indent=0):  # pragma: no cover
        """Return an indented string that describes the operation.

        This is mainly used for visualization of :class:`Circuit` instances.
        Notably, the returned string should not include the name of the
        operation, which would already be included by default in the tree
        string of the "parent" operation.

        It is fine to return an empty string, but otherwise, the string should
        be indented with `indent` copies of the string :data:`INDENT`, and
        terminated with a newline.

        :param indent: The amount of indent needed. Defaults to 0.
        :type indent: int, optional
        """
        return ""

    def __repr__(self) -> str:  # pragma: no cover
        return repr(vars(self))

    def __mul__(self, other):
        """Return two operations in parallel as a quantum circuit.

        The resulting :class:`Circuit` will have a number of qubits :math:`n`
        equal to the total number of qubits in both operands, indexed by
        integers from 0 to :math:`n-1`. The qubits are ordered naturally, i.e.,
        qubits in the left operand come first, and within both operands the
        original qubit orders are preserved.

        Note that this function will regard both operands as atomic operations:
        No attempt is made to "expand" the operands even if one or both of them
        are themselves :class:`Circuit` instances.
        """
        if isinstance(other, Operation):
            return Circuit().append(self, list(range(len(self.shape))), 0).append(
                other, list(range(len(self.shape), len(self.shape) + len(other.shape))), 0)
        else:
            return NotImplemented

    def __or__(self, other):
        """Return the concatenation of two operations as a quantum circuit.

        The left operand will happen first. Note that this ordering convention
        is different from how the result would be represented as a product of
        matrices. For example, ``ZeroState | HGate | ZGate`` will result in the
        state :math:`ZH|0\\rangle`.

        If the left operand is a :class:`Circuit` instance, then the qubit names
        in it will be preserved, and the same will apply to the right operand if
        it is a :class:`Circuit` instance too; otherwise qubits in the right
        operand will be indexed by integers from 0. The qubit names will
        determine how the two circuits are connected.

        If the left operand is not a :class:`Circuit` instance, then the qubits
        in *both* operands will be indexed by integers from 0, and the circuits
        will be connected correspondingly.
        """
        if isinstance(self, Circuit):
            if isinstance(other, Circuit):
                copy_self = copy(self)
                for time_step in other.operations_by_time:
                    for op in other.operations_by_time[time_step]:
                        operation = other.operations_by_time[time_step][op]
                        copy_self.append(operation['operation'],
                                         operation['qubits'])
                return copy_self
            else:
                return copy(self).append(other, list(range(len(other.shape))))
        else:
            return Circuit().append(self, list(range(len(self.shape)))).append(other, list(range(len(other.shape))))

    def _indices_with_property(self, pattern):
        res = [i for i in range(len(self.shape))
               if bool(re.match(pattern, self.shape[i]))]
        return res, len(res)

    @property
    def _input_indices(self):
        return self._indices_with_property("^[ibc]")

    @property
    def _output_indices(self):
        return self._indices_with_property(".*[odc]$")

    @property
    def tensor_pure(self):
        """Convert the operation into a tensor network representing the action of the operation in the pure state
        picture.

        Examples of tensor representations in the pure state picture include
        state vectors of pure states, and unitary matrices of unitary gates.

        Raises an error if the operation is not pure.
        """
        from .converter import Converter
        return Converter.convert_pure(self)

    @property
    def tensor_density(self):
        """Convert the operation into a tensor network representing the action of the operation in the density matrix
        picture.

        Examples of tensor representations in the density matrix picture include density matrices of general quantum
        states, and Choi matrices of quantum operations. As a special case, when the operation is pure, the tensor
        network returned by :attr:`tensor_density` will consist of two disjoint components, one being the tensor network
        returned by :attr:`tensor_pure` and the other being its adjoint.
        """
        from .converter import Converter
        return Converter.convert_density(self)

    @property
    def tensor_control(self):
        """Convert a controlled operation into a tensor network in the pure state picture, but with only one open edge
        for each controlling qubit.

        A qubit in an operation can be regarded as a controlling qubit if its
        value in the computational basis is never changed by the operation. As
        such, its input wire and output wire can be represented by the same edge
        in a tensor network, thus simplifying the tensor network. In other
        words, the :attr:`tensor_pure` for a controlled operation will be a
        block diagonal matrix, and its :attr:`tensor_control` will be a more
        compact representation of the same matrix.

        As a special case, if the operation is a diagonal gate, then every qubit
        can be regarded as a controlling qubit. See :class:`Diagonal`.
        """
        from .converter import Converter
        return Converter.convert_control(self)

    def adjoint(self):  # pragma: no cover
        """Return the adjoint of a quantum operation.

        ``~op`` is an alias of ``op.adjoint()``.
        """
        raise NotImplementedError()

    def __invert__(self):
        return self.adjoint()


class ImmutableOperation(Operation):
    """Class for quantum operations with explicit tensor representations. The operation is not supposed to be modified.

    :param data: A tensor representation of the operation. For this base class,
        it will be the tensor returned by :attr:`~Operation.tensor_density`.
        This means that for an operation with :math:`n` input qubits and
        :math:`m` output qubits, the tensor should be of rank :math:`2(n+m)`.
        A derived class of this may use different representation more suitable
        for a specific class of operations.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar ~.shape: A list of strings describing the input and output qubits of
        the quantum operation, with each string describing a qubit involved. For
        this base class, each string should be one of "i", "o" or "io",
        indicating whether each qubit is an input qubit, an output qubit, or
        both.
    :vartype ~.shape: List[str]

    :ivar name: Name of the quantum operation. Defaults to "ImOp".
    :vartype name: str, optional
    """

    def __init__(self,
                 data: np.ndarray,
                 shape: List[str],
                 name="ImOp") -> None:
        Operation.__init__(self, name)
        self.shape = shape
        self.process(data)

    def __str__(self) -> str:  # pragma: no cover
        return str(vars(self))

    def process(self, data):
        """Convert the input data into an appropriately shaped tensor, and initialize the operation with this tensor.

        A derived class that uses a different tensor representation of the
        operation should override this function in order to do shape checking
        and initialization correctly.

        :param data: The ``data`` parameter passed to
            :meth:`ImmutableOperation.__init__`.
        :type data: acqdp.tensor_network.tensor_valued.TensorValued or
            np.ndarray
        """
        _, lin = self._indices_with_property("^i")
        _, lout = self._indices_with_property(".*o$")
        if isinstance(data, TensorValued):
            if data.shape != tuple([2] * (2 * (lin + lout))):
                raise ValueError("Invalid operation: Input dimensions does not match the claimed shape")
            self._tensor_density = data
        elif type(data) != np.ndarray or np.size(data) != np.prod(np.shape(data)):
            raise TypeError("Invalid operation: Operation should be an np.ndarray.")
        else:
            try:
                self._tensor_density = Tensor(np.reshape(data, tuple([2] * (2 * (lin + lout)))))
            except ValueError:
                raise ValueError("Invalid operation: Operation dimension does not match qubits.")

    @classmethod
    def operation_from_kraus(cls, shape, kraus, name="ImOp_Kraus"):
        """Construct a quantum operation from its Kraus operator representation.

        :param shape: The shape of the operation, in the same format as the
            :attr:`shape` attribute of :class:`ImmutableOperation`.
        :type shape: List[str]

        :param kraus: The list of Kraus operators.
        :type kraus: List[acqdp.tensor_network.tensor_valued.TensorValued or
            np.ndarray]

        :param name: Name of the resulting quantum operation. Defaults to
            "ImOp_Kraus".
        :type name: str, optional

        :returns: A quantum operation constructed from the given Kraus operator
            representation.
        :rtype: ImmutableOperation
        """
        lin = len([i for i in shape if i[0] == 'i'])
        lout = len([i for i in shape if i[-1] == 'o'])
        shape_tensor = tuple([2] * (lin + lout))
        tensor_density = np.zeros(tuple([2] * (2 * (lin + lout))))
        for operator in kraus:
            if not isinstance(operator, np.ndarray):
                raise TypeError("Invalid operation: Kraus Operator should be an np.ndarray.")
            if (operator.shape != shape_tensor) and (operator.shape != (2 ** lout, 2 ** lin)):
                raise ValueError("Invalid operation: Dimensions do not match")
            i = np.reshape(operator, shape_tensor)
            tensor_density += np.multiply.outer(i, np.conj(i))
        return cls(tensor_density, shape, name)

    @property
    def is_pure(self):
        """Return True if the operation is a pure operation.

        Pure operations include pure states, isometries, projections and their
        combinations.

        Note that currently this function determines whether an operation is
        pure solely based on the class of the operation, without inspecting the
        actual data. For example, if a quantum state ``s`` is initialized with
        ``s = State(num_qubits, data)``, then ``s.is_pure`` will always be False
        and ``s.tensor_pure`` will always raise an error, even if ``data`` is
        actually the density matrix of a pure state.
        """
        return isinstance(self, PureOperation)


class State(ImmutableOperation):
    """Class for simple quantum states.

    Quantum states are regarded as a special case of quantum operations where
    there is no input qubit, and each qubit is an output qubit.

    :param num_qubits: Number of qubits in the state.
    :type num_qubits: int

    :param data: The density matrix representation of the state.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar name: Name of the quantum state. Defaults to "State".
    :vartype name: str, optional
    """

    def __init__(self,
                 num_qubits,
                 data: np.ndarray,
                 name="State") -> None:
        ImmutableOperation.__init__(self, data, tuple(['o'] * num_qubits), name)


class Measurement(ImmutableOperation):
    """Class for simple (destructive) measurements.

    Destructive measurements are regarded as a special case of quantum
    operations where each qubit is an input qubit, and there is no output
    qubit. Such a measurement maps an arbitrary quantum state to a number.

    :param num_qubits: Number of qubits measured.
    :type num_qubits: int

    :param data: The POVM (positive operator-valued measure) representation of
        the measurement.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar name: Name of the quantum state. Defaults to "Meas".
    :vartype name: str, optional
    """

    def __init__(self,
                 num_qubits,
                 data: np.ndarray,
                 name="Meas") -> None:
        ImmutableOperation.__init__(self, data, tuple(['i'] * num_qubits), name)


class Channel(ImmutableOperation):
    """Class for simple quantum channels.

    This class is used for the common case where the input and output Hilbert
    spaces are the same, i.e., each qubit is both an input qubit and an output
    qubit. For channels that do not satisfy this constraint, please use
    :class:`ImmutableOperation` directly.

    :param num_qubits: Number of qubits the channel operates on.
    :type num_qubits: int

    :param data: The tensor representation of the channel in the density matrix
        picture.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar name: Name of the quantum operation. Defaults to "Channel".
    :vartype name: str, optional
    """

    def __init__(self,
                 num_qubits,
                 data: np.ndarray,
                 name='Channel') -> None:
        ImmutableOperation.__init__(self, data, tuple(['io'] * num_qubits), name)


class PureOperation(ImmutableOperation):
    """Class for simple pure quantum operations.

    :param data: The tensor representation of the operation in the pure state
        picture. It will be the tensor returned by
        :attr:`~Operation.tensor_pure`. This means that for an operation with
        :math:`n` input qubits and :math:`m` output qubits, the tensor should be
        of rank :math:`n+m`, i.e., half the rank of the tensor in the density
        matrix picture.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar ~.shape: The shape of the operation, in the same format as the
        :attr:`shape` attribute of :class:`ImmutableOperation`.
    :vartype ~.shape: List[str]

    :ivar name: Name of the quantum operation. Defaults to "PureOp".
    :vartype name: str, optional

    :ivar self_adjoint: Whether the operation is self-adjoint. Defaults to
        False.
    :vartype self_adjoint: bool, optional
    """

    def __init__(self,
                 data: np.ndarray,
                 shape: List[str],
                 name="PureOp",
                 self_adjoint=False) -> None:
        ImmutableOperation.__init__(self, data, shape, name)
        self.self_adjoint = self_adjoint

    def set_adjoint_op(self, adjoint_op):
        """Set the known adjoint of the operation.

        Usually, the adjoint of a pure operation is constructed on demand by
        calculating the :attr:`~Operation.tensor_pure` of the operation, then
        constructing a new :class:`PureOperation` object, which can be an
        inefficient process. By setting the adjoint of a operation to a known
        value, one can bypass this procedure. Note that, for efficiency, there
        is no check that ``adjoint_op`` is actually the adjoint of ``self``.

        The adjoint of ``adjoint_op`` is also set to ``self`` so there is no
        need to use this function twice.

        :param adjoint_op: The known adjoint of ``self``.
        :type adjoint_op: PureOperation
        """
        self.adjoint_op = adjoint_op
        adjoint_op.adjoint_op = self

    def process(self, data):
        _, lin = self._indices_with_property("^i")
        _, lout = self._indices_with_property(".*o$")
        if isinstance(data, TensorValued):
            if data.shape != tuple([2] * (lin + lout)):
                raise ValueError("Invalid operation: Input dimensions does not match the claimed shape")
            self._tensor_pure = data
        elif type(data) != np.ndarray or np.size(data) != np.prod(np.shape(data)):
            raise TypeError("Invalid operation: Operation should be an np.ndarray.")
        else:
            try:
                self._tensor_pure = Tensor(np.reshape(data, tuple([2] * (lin + lout))))
            except ValueError:
                raise ValueError("Invalid operation: Operation dimension does not match qubits.")

    def adjoint(self):
        if self.self_adjoint:
            return self
        elif hasattr(self, 'adjoint_op'):
            return self.adjoint_op
        else:
            transition_dict = {'i': 'o', 'o': 'i'}
            shape = ["".join([transition_dict[i] for i in s[::-1]]) for s in self.shape]
            _, lin = self._indices_with_property("^i")
            _, lout = self._indices_with_property(".*o$")
            tensor_pure = (~self._tensor_pure) % tuple(list(range(lout, lout + lin)) + list(range(lout)))
            return PureOperation(tensor_pure, shape, "~" + self.name)


class PureState(PureOperation, State):
    """Class for simple pure quantum states.

    :param num_qubits: Number of qubits in the state.
    :type num_qubits: int

    :param data: The state vector representation of the state.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar name: Name of the quantum state. Defaults to "PureState".
    :vartype name: str, optional
    """

    def __init__(self,
                 num_qubits,
                 data: np.ndarray,
                 name='PureState') -> None:
        PureOperation.__init__(self, data, tuple(['o'] * num_qubits), name, False)


class PureMeas(PureOperation, Measurement):
    """Class for simple projective measurements.

    :param num_qubits: Number of qubits measured.
    :type num_qubits: int

    :param data: The vector representation of the measurement. Note that it
        should be the *complex conjugation* of the state vector of the state
        projected onto.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar name: Name of the quantum state. Defaults to "PureMeas".
    :vartype name: str, optional
    """

    def __init__(self,
                 num_qubits,
                 data: np.ndarray,
                 name='PureMeas') -> None:
        PureOperation.__init__(self, data, tuple(['i'] * num_qubits), name, False)


class Unitary(PureOperation, Channel):
    """Class for simple unitary gates.

    :param num_qubits: Number of qubits the unitary operates on.
    :type num_qubits: int

    :param data: The matrix representation of the unitary.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar name: Name of the quantum state. Defaults to "Unitary".
    :vartype name: str, optional

    :ivar self_adjoint: Whether the unitary is self-adjoint. Defaults to False.
    :vartype self_adjoint: bool, optional
    """

    def __init__(self,
                 num_qubits,
                 data: np.ndarray,
                 name='Unitary',
                 self_adjoint=False) -> None:
        PureOperation.__init__(self, data, tuple(['io'] * num_qubits), name, self_adjoint)


class ControlledOperation(PureOperation):
    """Class for simple controlled operations.

    :param data: The tensor representation of the controlled operation. It will
        be the tensor returned by :attr:`~Operation.tensor_control`. This means
        that for an operation with :math:`k` controlling qubits, :math:`n`
        non-control input qubits and :math:`m` non-control output qubits, the
        tensor should be of rank :math:`k+n+m`.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar ~.shape: The shape of the operation, in the same format as the
        :attr:`shape` attribute of :class:`ImmutableOperation`, but in addition
        to "i", "o", and "io", the string "c" is also allowed, which indicates a
        controlling qubit.
    :vartype ~.shape: List[str]

    :ivar name: Name of the quantum operation. Defaults to "C-Op".
    :vartype name: str, optional

    :ivar self_adjoint: Whether the operation is self-adjoint. Defaults to
        False.
    :vartype self_adjoint: bool, optional
    """

    def __init__(self,
                 data: np.ndarray,
                 shape: List[str],
                 name="C-Op",
                 self_adjoint=False) -> None:
        PureOperation.__init__(self, data, shape, name, self_adjoint)

    def process(self, data):
        _, lctrl = self._indices_with_property("^c$")
        _, lin = self._indices_with_property("^i")
        _, lout = self._indices_with_property(".*o$")
        if isinstance(data, TensorValued):
            if data.shape != tuple([2] * (lctrl + lin + lout)):
                raise ValueError("Invalid operation: Input dimensions does not match the claimed shape")
            self._tensor_control = data
        elif type(data) != np.ndarray or np.size(data) != np.prod(np.shape(data)):
            raise TypeError("Invalid operation: Operation should be an np.ndarray.")
        else:
            try:
                self._tensor_control = Tensor(np.reshape(data, tuple([2] * (lctrl + lin + lout))))
            except ValueError:
                raise ValueError("Invalid operation: Operation dimension does not match qubits.")

    def adjoint(self):
        if self.self_adjoint:
            return self
        else:
            transition_dict = {'i': 'o', 'o': 'i', 'c': 'c'}
            shape = ["".join([transition_dict[i] for i in s[::-1]]) for s in self.shape]
            _, lctrl = self._indices_with_property("^c$")
            _, lin = self._indices_with_property("^i")
            _, lout = self._indices_with_property(".*o$")
            tensor_control = ~(self._tensor_control)\
                % tuple(list(range(lctrl))
                        + list(range(lctrl + lout, lctrl + lout + lin))
                        + list(range(lctrl, lctrl + lout)))
            return ControlledOperation(tensor_control, shape, "~" + self.name)


class Diagonal(ControlledOperation):
    """Class for simple diagonal gates.

    A diagonal gate can be regarded as a controlled phase shift, where every
    qubit is a controlling qubit.

    :param num_qubits: Number of qubits the diagonal gate operates on.
    :type num_qubits: int

    :param data: The diagonal elements of the matrix representation of the gate.
    :type data: acqdp.tensor_network.tensor_valued.TensorValued or np.ndarray

    :ivar name: Name of the quantum state. Defaults to "Diag".
    :vartype name: str, optional

    :ivar self_adjoint: Whether the gate is self-adjoint. Defaults to False.
    :vartype self_adjoint: bool, optional
    """

    def __init__(self,
                 num_cbits,
                 data: np.ndarray,
                 name='Diag',
                 self_adjoint=False) -> None:
        ControlledOperation.__init__(self, data, tuple(['c'] * num_cbits), name, self_adjoint)


XGate = Unitary(1, np.array([[0, 1], [1, 0]]), "X", True)
"""Single-qubit Pauli X gate."""

YGate = Unitary(1, np.array([[0, 1j], [-1j, 0]]), "Y", True)
"""Single-qubit Pauli Y gate."""

ZGate = Diagonal(1, np.array([1, -1]), "Z", True)
"""Single-qubit Pauli Z gate."""

HGate = Unitary(1, np.sqrt(0.5) * np.array([[1, 1], [1, -1]]), "H", True)
"""Single-qubit Hadamard gate."""

TGate = Diagonal(1, np.array([1, (1 + 1j) * np.sqrt(0.5)]), "T")
"""Single-qubit T gate, i.e., a :math:`\\pi/4` rotation around the Z axis on the Bloch sphere."""

SGate = Diagonal(1, np.array([1, 1j]), "S")
"""Single-qubit S gate, i.e., a :math:`\\pi/2` rotation around the Z axis on the Bloch sphere."""

Trace = Measurement(1, TensorNetwork(open_edges=[0, 0], bond_dim=2), "Tr")
"""The partial trace operation, defined as an measurement that maps every
normalized single-qubit state to 1.

In a circuit, a partial trace operation can simulate discarding a qubit. Note
that this operation is inherently not pure since discarding one part of an
entangled pure state will result in a mixed state.
"""


IGate = Unitary(1,
                TensorNetwork(open_edges=[0, 0], bond_dim=2),
                "I", True)
"""Single-qubit Identity gate."""


SWAPGate = Unitary(2,
                   TensorNetwork(open_edges=[0, 1, 1, 0], bond_dim=2),
                   "SWAP", True)
"""Two-qubit SWAP gate."""

CZGate = Diagonal(2, np.array([[1, 1], [1, -1]]), name='CZ', self_adjoint=True)
"""Two-qubit CZ gate."""

XHalfGate = Unitary(1,
                    np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) * 0.5,
                    "X/2")
"""Single-qubit :math:`\\sqrt{X}` gate."""

YHalfGate = Unitary(1,
                    np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) * 0.5,
                    "Y/2")
"""Single-qubit :math:`\\sqrt{Y}` gate."""

CNOTGate = ControlledOperation(np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),
                               ['c', 'io'],
                               "CNOT", True)
"""Two-qubit CNOT gate."""

ZeroState = PureState(1, np.array([1, 0]), '|0>')
"""Single-qubit state :math:`|0\\rangle`."""

OneState = PureState(1, np.array([0, 1]), '|1>')
"""Single-qubit state :math:`|1\\rangle`."""

CompState = [ZeroState, OneState]
"""List of the computational basis states, :math:`|0\\rangle` and :math:`|1\\rangle`."""

PlusState = PureState(1, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), '|+>')
"""Single-qubit state :math:`|+\\rangle`."""

MinusState = PureState(1, np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]), '|->')
"""Single-qubit state :math:`|-\\rangle`."""

FourierState = [PlusState, MinusState]
"""List of the Hadamard basis states, :math:`|+\\rangle` and :math:`|-\\rangle`."""

ZeroMeas = PureMeas(1, np.array([1, 0]), '<0|')
"""Single-qubit measurement :math:`\\langle 0|`, the adjoint of :math:`|0\\rangle`."""

OneMeas = PureMeas(1, np.array([0, 1]), '<1|')
"""Single-qubit measurement :math:`\\langle 1|`, the adjoint of :math:`|1\\rangle`."""

CompMeas = [ZeroMeas, OneMeas]
"""List of the computational basis measurements, :math:`\\langle 0|` and :math:`\\langle 1|`."""

PlusMeas = PureMeas(1, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), '<+|')
"""Single-qubit measurement :math:`\\langle +|`, the adjoint of :math:`|+\\rangle`."""

MinusMeas = PureMeas(1, np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]), '<-|')
"""Single-qubit measurement :math:`\\langle -|`, the adjoint of :math:`|-\\rangle`."""

FourierMeas = [PlusMeas, MinusMeas]
"""List of the computational basis measurements, :math:`\\langle +|` and :math:`\\langle -|`."""

NDCompMeas = Channel(1, TensorNetwork([0, 0, 0, 0], bond_dim=2), 'ND')
"""A non-destructive computational basis measurement.

This is equivalent to :class:`acqdp.circuit.noise.Dephasing()`, but with a
slightly more efficient tensor network representation.
"""

ZeroState.set_adjoint_op(ZeroMeas)
OneState.set_adjoint_op(OneMeas)
PlusState.set_adjoint_op(PlusMeas)
MinusState.set_adjoint_op(MinusMeas)


class XRotation(Unitary):
    """Class for X Rotation gates in the form of :math:`e^{i\\theta X}`.

    Note that, since the eigenvalues of the Pauli operator :math:`X` are 1 and
    -1, this actually corresponds to a rotation by :math:`2\\theta` on the Bloch
    sphere. For example, ``XRotation(np.pi)`` is a global phase operation with
    no physical effect, equivalent to a rotation by :math:`2\\pi` on the Bloch
    sphere. On the other hand, ``XRotation(np.pi / 2)`` is equivalent to the
    Pauli X operator up to a global phase, i.e., a rotation by :math:`\\pi`
    around the X axis on the Bloch sphere.

    :ivar angle: The rotation angle :math:`\\theta`.
    :vartype angle: float
    """

    def __init__(self, angle):
        self.angle = angle
        Unitary.__init__(self, 1,
                         np.array([[np.cos(angle), 1j * np.sin(angle)],
                                   [1j * np.sin(angle), np.cos(angle)]]),
                         "R_X")

    def adjoint(self):
        return XRotation(-self.angle)


class ZRotation(Diagonal):
    """Class for Z Rotation gates in the form of :math:`e^{i\\theta Z}`.

    The same note about :class:`XRotation` applies here. Furthermore, note that
    the rotation direction is the reverse of the "natural" direction given by
    :math:`\\mathop{\\mathrm{diag}}(1, e^{i\\theta})`, since the actual rotation
    is :math:`\\mathop{\\mathrm{diag}}(e^{i\\theta}, e^{-i\\theta})`. For
    example, ``ZRotation(-np.pi / 8)`` and ``ZRotation(-np.pi / 4)`` will give
    the T gate and the S gate respectively, up to global phases.

    :ivar angle: The rotation angle :math:`\\theta`.
    :vartype angle: float
    """

    def __init__(self, angle):
        self.angle = angle
        Diagonal.__init__(self, 1,
                          np.array([np.exp(1j * angle), np.exp(-1j * angle)]),
                          "R_Z({})".format(angle))

    def adjoint(self):
        return ZRotation(-self.angle)


class Circuit(Operation):
    """Class for quantum circuits that can be manipulated by adding and removing operations.

    Each operation in a circuit has a name that is unique within the circuit,
    not necessarily related to its own :attr:`name` attribute. Each operation
    also has a time step which determines the order of the operations. For
    applications where those do not matter, they can be kept at their default
    values; see :meth:`append` for those default values.

    A circuit is always empty when initialized.

    :ivar name: Name of the quantum circuit. Defaults to "Circuit".
    :vartype name: str, optional

    :ivar operations_by_name: A dict mapping from operation names to dicts
        describing operations in the circuit. Each of the latter dicts ``d`` has
        three entries:

        * ``d["operation"]`` is the actual :class:`Operation`.
        * ``d["time_step"]`` gives the time step at which the opration happens.
        * ``d["qubits"]`` is a list of qubits indicating which qubits in the
          circuit the operation is applied to.
    :vartype operations_by_name: Dict[hashable, dict]
    """

    def __init__(self, name='Circuit') -> None:
        Operation.__init__(self, name)
        self.operations_by_name = {}

    @property
    def max_time(self) -> float:
        """Return the maximum time step among all operations in the circuit."""
        if(len(self.operations_by_name) == 0):
            return -1
        return max(map(lambda x: x["time_step"], self.operations_by_name.values()))

    def append(self,
               operation: Operation,
               qubits: List[int],
               time_step: Optional[float] = None,
               name=None) -> None:
        """Add an operation to the circuit.

        If a time step is given, the operation is inserted into the circuit at
        the given time. Otherwise, the operation is appended to the end of the
        circuit.

        :param operation: The operation to add to the circuit.
        :type operation: Operation

        :param qubits: The qubits the operation is applied to. Qubit names can
            be anything as long as they are hashable and comparable with each
            other. Any qubits not already in the circuit are added to the
            circuit.
        :type qubits: list

        :param time_step: The time step at which the operation happens. Defaults
            to :attr:`max_time` plus one, which ensures that the operation is
            appended to the end of the cirucit.
        :type time_step: int or float, optional

        :param name: The name of the operation in the circuit. By default, the
            name is a tuple generated from ``time_step`` and ``qubits``,
            ensuring its uniqueness within the circuit.
        :type name: hashable, optional
        """
        if time_step is None:
            time_step = 1 + self.max_time
        if name is None:
            name = (time_step,
                    tuple([q for q in qubits]))
        if (len(qubits) != len(set(qubits))
                or len(qubits) != len(operation.shape)):
            raise ValueError("Invalid qubits: qubits not match operation.")
        self.operations_by_name[name] = {"operation": operation,
                                         "time_step": time_step,
                                         "qubits": qubits}
        return self

    @property
    def operations_by_time(self):
        """Return an :class:`OrderedDict` of dicts, which is an reorganization of :attr:`operations_by_name` such that
        ``c.operations_by_time[t][name]== c.operations_by_name[name]``, where ``t`` is the time step of the
        operation in question.

        The returned ``OrderedDict`` in ordered by increasing time.
        """
        time_steps = set(i["time_step"] for i in self.operations_by_name.values())
        d = {t: {} for t in time_steps}
        for g in self.operations_by_name:
            d[self.operations_by_name[g]["time_step"]][g] = self.operations_by_name[g]
        return collections.OrderedDict(sorted(d.items()))

    def tree_string(self, indent=0) -> str:  # pragma: no cover
        indentation = indent * INDENT
        s = indentation + "Ops:\n"
        d = self.operations_by_time
        for t in d:
            for n in d[t]:
                s += indentation + INDENT + str(n) + "\n"
                s += d[t][n]['operation'].tree_string(indent + 1)
        s += indentation + "End of circuit\n"
        return s

    @property
    def all_qubits(self) -> List[int]:
        """Return a list of all qubits in the circuit."""
        s = []
        for g in self.operations_by_name:
            s += self.operations_by_name[g]["qubits"]
        try:
            return sorted(set(s))
        except TypeError:
            dic = {str(a): a for a in s}
            return [dic[t] for t in sorted(dic)]

    @property
    def shape(self) -> List[str]:
        """Return the shape of the circuit as an operation, in the same format
        as the :attr:`~ImmutableOperation.shape` attribute of
        :class:`ImmutableOperation`.

        Raises an :class:`ValueError` if the circuit is invalid. This can happen
        if a qubit is used multiple times at the same time step, or if the
        shapes of adjacent operations on the same qubit do not fit together,
        such as a quantum state followed by another quantum state.
        """
        shape = {x: "" for x in self.all_qubits}
        trans_dict = {'b': 'i', 'd': 'o', 'c': 'io', 'i': 'i', 'o': 'o'}
        last_time_used = {x: -math.inf for x in self.all_qubits}
        for time_step in self.operations_by_time:
            for gate_name in self.operations_by_time[time_step]:
                gate = self.operations_by_name[gate_name]
                for i, qubit in enumerate(gate['qubits']):
                    if last_time_used[qubit] >= time_step:
                        raise ValueError(
                            "Invalid connection: Qubit {} used multiple times at timestep {}".format(qubit, time_step))
                    last_time_used[qubit] = time_step
                    shape[qubit] += "".join([trans_dict[letter] for letter in gate['operation'].shape[i]])
        pattern = re.compile("^o?(io)*i?$")
        for qubit in shape:
            if pattern.match(shape[qubit]):
                shape[qubit] = shape[qubit][: 2 - len(shape[qubit]) % 2]
            else:
                raise ValueError("Invalid connection on qubit {}".format(qubit))
        return tuple(shape[qubit] for qubit in sorted(shape))

    @property
    def is_valid(self) -> bool:
        """Return True if the circuit is valid.

        See :attr:`shape` for ways a circuit can be invalid.
        """
        try:
            self.shape
            return True
        except ValueError:
            return False

    @property
    def is_pure(self) -> bool:
        """Return True if the operation is a pure operation.

        Pure operations include pure states, isometries, projections and their
        combinations.

        Similar to :attr:`ImmutableOperation.is_pure`, this does not analyze the
        actual action of the circuit, and only checks whether the component
        operations are all pure.
        """
        return self.is_valid and np.all([x["operation"].is_pure for x in self.operations_by_name.values()])

    def adjoint(self):
        if not self.is_pure:
            raise ValueError("ValueError: Noisy circuit does not have an adjoint")
        c = Circuit()
        for operation_name in self.operations_by_name:
            operation = self.operations_by_name[operation_name]['operation']
            qubits = self.operations_by_name[operation_name]['qubits']
            time_step = self.max_time - self.operations_by_name[operation_name]['time_step']
            c.append(~operation, qubits, time_step)
        return c

    def __copy__(self):
        c = Circuit()
        for operation_name in self.operations_by_name:
            operation = self.operations_by_name[operation_name]['operation']
            qubits = self.operations_by_name[operation_name]['qubits']
            time_step = self.operations_by_name[operation_name]['time_step']
            c.append(operation, qubits, time_step, operation_name)
        return c


class ControlledCircuit(ControlledOperation):
    """Class for quantum circuits controlled by a single qubit.

    :ivar ~_.circuit: The circuit being controlled.
    :vartype ~_.circuit: Circuit

    :ivar name: Name of the controlled circuit. Defaults to "C-<circuit>", where
        <circuit> is the name of the circuit being controlled.
    :vartype name: str, optional

    :ivar conditioned_on: Whether the circuit is applied when the controlling
        qubit is in the :math:`|1\\rangle` state or the :math:`|0\\rangle`
        state. Defaults to True, meaning that the circuit is applied when the
        controlling qubit is in the :math:`|1\\rangle` state.
    :vartype conditioned_on: bool, optional
    """

    def __init__(self,
                 circuit,
                 name=None,
                 conditioned_on=True):
        if name is None:
            name = "C-" + circuit.name
        self.shape = tuple(['c'] + list(circuit.shape))
        self.circuit = circuit
        self.conditioned_on = conditioned_on
        self.name = name

    def tree_string(self, indent=0):  # pragma: no cover
        indentation = indent * INDENT
        s = indentation + "Conditioned on:{}\n".format(self.conditioned_on)
        s += self.circuit.tree_string(indent + 1)
        s += indentation + "End of control\n"
        return s

    def adjoint(self):
        return ControlledCircuit(~(self.circuit), conditioned_on=self.conditioned_on)

    @property
    def is_valid(self):
        """Return True if the circuit is valid.

        See :attr:`Circuit.shape` for ways a circuit can be invalid.
        """
        return self.circuit.is_valid


def Controlled(operation, name=None, conditioned_on=True):
    """Return a controlled version of the given operation.

    When the given operation is not applied, the identity gate is applied, which
    means the operation must have the same input and output qubits. The
    operation also needs to be pure in order for quantum control to be
    well-defined, which means it must be a unitary. However, it can be either
    an :class:`ImmutableOperation` or a :class:`Circuit`.

    :param operation: The quantum operation being controlled.
    :type operation: Operation

    :param name: Name of the resulting controlled operation. Defaults to
        "C-<operation>", where <operation> is the operation being controlled.
    :type name: str, optional

    :param conditioned_on: Whether the operation is applied when the controlling
        qubit is in the :math:`|1\\rangle` state or the :math:`|0\\rangle`
        state. Defaults to True, meaning that the operation is applied when the
        controlling qubit is in the :math:`|1\\rangle` state.
    :type conditioned_on: bool, optional
    """
    if name is None:
        name = "C{}-".format("" if conditioned_on else 0) + str(operation.name)
    if not (operation.is_pure and all(x == 'io' or x == 'c' for x in operation.shape)):
        raise ValueError("Controlled unitary must be unitary")
    shape = tuple(['c'] + list(operation.shape))
    len_qubits = len(operation.shape)
    _, len_cbits = operation._indices_with_property("^c$")
    if isinstance(operation, Circuit):
        return ControlledCircuit(operation, name, conditioned_on)
    else:
        data = operation.tensor_control.contract()
        tn = TensorNetwork()
        for i in range(len_qubits):
            tn.open_edge(i, 2)
        for i in range(len_cbits, len_qubits):
            tn.open_edge(i)
        identity = tn.contract()
        if conditioned_on:
            return ControlledOperation(np.stack((identity, data)), shape, name)
        else:
            return ControlledOperation(np.stack((data, identity)), shape, name)


class SuperPosition(Operation):
    """Class for a superposition of multiple pure operations.

    :ivar operations: A list of operations in the superposition.
    :vartype operations: List[Operation]

    :ivar coefs: A list of coefficients, corresponding to each of the component
        operations. By default, all the coefficients are 1.
    :vartype coefs: List[float], optional

    :ivar name: Name of the superposition operation. Defaults to "SuperPos".
    :vartype name: str, optional
    """

    def __init__(self,
                 operations,
                 coefs=None,
                 name='SuperPos'):
        if coefs is None:
            coefs = np.ones(len(operations))
        if len(coefs) != len(operations):
            raise ValueError("Invalid operation: length of components does not match length of coefficients")
        Operation.__init__(self, name)
        input_indices = operations[0]._input_indices
        output_indices = operations[0]._output_indices
        shape = list(operations[0].shape)
        for op in operations:
            if (op._input_indices != input_indices)\
               or (op._output_indices != output_indices):
                raise ValueError("Invalid operation: components have incompatible shape")
            if not op.is_pure:
                raise ValueError("Invalid superposition: all components must be pure")
            for s in range(len(op.shape)):
                if shape[s] == 'c':
                    shape[s] = 'io'
        self.coefs = coefs
        self.operations = operations
        self.shape = tuple(shape)

    @property
    def is_pure(self):
        """Return True if the operation is a pure operation.

        A :class:`SuperPosition` object is always pure.
        """
        return True

    def adjoint(self):
        return SuperPosition([~op for op in self.operations],
                             np.conj(self.coefs),
                             "~" + str(self.name))


def XXRotation(angle):
    """Return a two-qubit XX rotation gate in the form of
    :math:`e^{i\\theta(X\\otimes X)}`, also known as an Ising gate.

    :param angle: The rotation angle :math:`\\theta`.
    :type angle: float
    """
    return SuperPosition([IGate * IGate, XGate * XGate],
                         [np.cos(angle), np.sin(angle) * 1j],
                         'R_XX({})'.format(angle))


def ZZRotation(angle):
    """Return a two-qubit ZZ rotation gate in the form of
    :math:`e^{i\\theta(Z\\otimes Z)}`.

    :param angle: The rotation angle :math:`\\theta`.
    :type angle: float
    """
    return Diagonal(2,
                    np.array([[np.exp(1j * angle), np.exp(-1j * angle)],
                              [np.exp(-1j * angle), np.exp(1j * angle)]]),
                    'R_ZZ({})'.format(angle))
