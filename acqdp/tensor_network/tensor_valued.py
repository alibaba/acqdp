import numpy

DTYPE = complex


def conjugate(t) -> 'TensorValued':
    """Return the complex conjugation of a `TensorValued` object.

    :returns:  :class:`TensorValued` -- the conjugation of a `TensorValued` object.
    """
    from .tensor import Tensor
    from .tensor_sum import TensorSum
    from .tensor_network import TensorNetwork
    from .tensor_view import TensorView
    if not isinstance(t, TensorValued):
        raise TypeError()
    elif isinstance(t, Tensor):
        if t._data is None:
            return Tensor()
        else:
            return Tensor(numpy.conj(t._data))
    elif isinstance(t, TensorSum):
        ts = t.copy()
        for term in ts.terms_by_name:
            ts.terms_by_name[term] = conjugate(ts.terms_by_name[term])
        return ts
    elif isinstance(t, TensorNetwork):
        tn = t.copy()
        for node in tn.nodes:
            tn.network.nodes[node]['tensor'] = conjugate(tn.network.nodes[node]['tensor'])
        return tn
    elif isinstance(t, TensorView):
        return TensorView(t)


def transpose(t, axes: tuple) -> 'TensorValued':
    """Return the transposition of a `TensorValued` object.

    :param axes: the transposition on the referred object.
    :type axes: tuple.
    :returns:  :class:`TensorNetwork` -- the transposition of a `TensorValued` object can be readily expressed in terms of a
        `TensorNetwork` object.
    """
    from .tensor_network import TensorNetwork
    tn = TensorNetwork()
    tn.add_node(tensor=t, is_open=True)
    tn.open_edges = [tn.open_edges[i] for i in axes]
    return tn


def normalize(tv: 'TensorValued') -> 'TensorValued':
    """Return a copy with unit Frobenius norm.

    :param tv: The :class:`TensorValued` object to normalize.
    :type tv: :class:`TensorValued`.
    :returns:  :class:`TensorValued` -- the normalized :class:`TensorValued` object.
    """
    return tv * (1 / numpy.sqrt(tv.norm_squared))


class TensorValued(object):
    """Interface for all :class:`TensorValued` objects, including :class:`Tensor`, :class:`TensorNetwork`,
    :class:`TensorSum` and :class:`TensorView`.

    :ivar identifier: unique identifier for each :class:`TensorValued` object.
    :ivar dtype: A :class:`TensorValued` object is homogeneous, and contains elements described by a dtype object.
    """
    id_count = -1

    def __new__(cls, *args, **kwargs):
        TensorValued.id_count += 1
        instance = object.__new__(cls)
        return instance

    def __init__(self, dtype: type = DTYPE) -> None:
        self.identifier = TensorValued.id_count
        self.dtype = dtype

    @property
    def shape(self):
        """The common property of all :class:`TensorValued` classes, indicating whether the bond dimensions for a tensor
        valued object. A tensor valued object is semantically a multi-dimensionally array, and its dimensions can be
        expressed as a tuple of integers. The tuple is called the shape of the tensor, whereas the length of the tuple
        is the rank of the tensor.

        In ACQDP, undetermined tensors and undetermined dimensions are allowed. In the former case, the shape of the tensor
        will return `None`; in the latter case, some of the bond_dimensions appearing in the tuple could be `None`.

        :returns: :class:`tuple` or None
        :raises: NotImplementedError, ValueError
        """
        raise NotImplementedError()

    @property
    def is_valid(self) -> bool:
        """The common property of all :class:`TensorValued` classes, indicating whether the :class:`TensorValued` object
        is valid or not. In every step of a program, all existing :class:`TensorValued` object must be valid, otherwise
        an exception should be thrown out.

        :returns: :class:`bool`
        :raises: NotImplementedError
        """
        raise NotImplementedError()

    @property
    def is_ready(self) -> bool:
        """The common property of all :class:`TensorValued` classes, indicating whether the current
        :class:`TensorValued` object is ready for contraction, i.e. whether it semantically represents a tensor with a
        definite value. A `TensorValued` object needs to be ready upon contraction, but needs not to be ready throught
        the construction.

        :returns: :class:`bool`
        :raises: NotImplementedError()
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return "Type: " + str(type(self))\
               + ", Shape: " + str(self.shape)

    def __repr__(self) -> str:
        return "Id: " + str(self.identifier) + self.__str__()

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        elif isinstance(other, TensorValued):
            if self.shape != other.shape:
                return False
            return numpy.allclose(self.contract(), other.contract())
        else:
            return False

    def __mul__(self, other) -> 'TensorValued':
        """Tensor product of two tensors. For tensor multiplications, use `TensorNetwork` classes to specify how indices
        are to be contracted. By default, addition creates a `TensorNetwork` object with two operands as two components.

        :returns: :class:`TensorNetwork`
        """
        from .tensor_network import TensorNetwork
        tn = TensorNetwork()
        tn.add_node(tensor=self, is_open=True)
        tn.add_node(tensor=other, is_open=True)
        return tn

    __rmul__ = __mul__

    def __add__(self, other: 'TensorValued') -> 'TensorValued':
        """Addition of two `TensorValued` objects.

        By default, addition creates a `TensorSum` object with two operands as two components. The two operands of the
        addition must have compatible shapes.
        """
        from .tensor_sum import TensorSum
        tl = TensorSum()
        tl.add_term(tensor=self)
        tl.add_term(tensor=other)
        return tl

    def __neg__(self) -> 'TensorValued':
        """Negation of a `TensorValued` object."""
        return -1 * self

    def __sub__(self, other: 'TensorValued') -> 'TensorValued':
        """Subtraction of `TensorValued` objects."""
        return self + (-other)

    def __invert__(self) -> 'TensorValued':
        """Syntactic sugar for conjugation."""
        return conjugate(self)

    def __mod__(self, axes: tuple):
        """Syntactic sugar for transposition."""
        return transpose(self, axes)

    @property
    def norm_squared(self):
        """Square of Frobenius norm of the :class:`TensorValued` object."""
        from .tensor_network import TensorNetwork
        tn = TensorNetwork()
        tn.add_node(tensor=self)
        tn.add_node(tensor=~self)
        return tn.contract().real

    def fix_index(self, index, fix_to=0) -> 'TensorValued':
        """Fix the given index to the given value. The result :class:`TensorValued` object would have the same type as
        the original one, with rank 1 smaller than the original.

        :param index: The index to fix.
        :type index: :class:`int`
        :param fix_to: the value to assign to the given index.
        :type fix_to: :class:`bool`
        :returns:  :class:`TensorValued` -- The :class:`TensorValued` object after fixing the given index.
        :raises: NotImplementedError
        """
        raise NotImplementedError()

    def expand(self):
        """Expand nested tensor network structures in the :class:`TensorValued` object if there is any.

        :returns: :class:`TensorValued`.
        """
        return self

    def contract(self, **kwargs) -> numpy.ndarray:
        """Evaluate the :class:`TensorValued` object to a :class:`numpy.ndarray`.

        :returns: :class:`numpy.ndarray`
        :raises: NotImplementedError
        """
        raise NotImplementedError()

    def cast(self, dtype):
        """Cast the tensor valued object to a new underlying dtype."""
        self.dtype = dtype

    def copy(self) -> 'TensorValued':
        """Make a copy of the current object.

        Data is duplicated only when necessary.
        """
        raise NotImplementedError()

    def __deepcopy__(self, memo) -> 'TensorValued':
        """Make a deepcopy of the current object with all data duplicated."""
        raise NotImplementedError()
