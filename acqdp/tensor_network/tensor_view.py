import copy
import numpy
from .tensor_valued import TensorValued, DTYPE


class TensorView(TensorValued):
    """
    :class:`TensorView` is a subclass of :class:`TensorValued` representing unary operations over another :class:`TensorValued`
        object that preserves the shape of the tensor. Common examples include element-wise conjugation and normalization with
        respect to the frobenius norm.

    :ivar tn: the underlying `TensorValued` object where the unary operation is performed onto.
    :ivar func: the unary function to be applied.
    :ivar homomorphism: indicator whether the unary function is homomorphic to the addition and multiplication of tensors. If
        so, the unary function can be broadcast to lower-level tensors, enabling potential simplification of the tensor
        network structure.
    :ivar dtype: dtype for the tensor entries.
    """

    def __init__(self, tn, func=numpy.conj, homomorphism=False, dtype=DTYPE):
        """The constructor of a `TensorView` object."""
        super().__init__(dtype)
        self.ref = tn
        self.func = func
        self.homomorphism = (func == numpy.conj) | homomorphism

    def __str__(self) -> str:
        data_str = "Data: \n" + str(self.ref)
        func_str = "Func: " + str(self.func)
        return super().__str__() + "\n" + data_str + func_str

    @property
    def shape(self):
        """
        The common property of all :class:`TensorValued` classes, yielding the shape of the object.
        :class:`TensorValued` objects must have compatible shapes in order to be connected together in
            a :class:`TensorNetwork`, or summed over in a :class:`TensorSum`.

        For :class:`TensorView` objects, it refers to the shape of the underlying :class:`TensorValued` object where the unary
            operation is performed onto.
        """
        return self.ref.shape

    @property
    def is_ready(self):
        """The common property of all :class:`TensorValued` classes, indicating whether the current
        :class:`TensorValued` object is ready for contraction, i.e. whether it semantically represents a tensor with a
        definite value. In the process of a program, not all :class:`TensorValued` objects need to be ready; however
        once the `data` property of a certain object is queried, such object must be ready in order to successfully
        yield an :class:`numpy.ndarray` object.

        For :class:`TensorView` objects, it is to indicate whether the underlying :class:`TensorValued` object where the
        unary operation is performed onto, is ready for contraction.
        """
        return self.ref.is_ready

    @property
    def is_valid(self):
        """The common property of all :class:`TensorValued` classes, indicating whether the :class:`TensorValued` object
        is valid or not. In every step of a program, all existing :class:`TensorValued` object must be valid, otherwise
        an exception should be thrown out; this property is for double checking that the current :class:`TensorValued`
        object is indeed valid.

        For :class:`TensorView` objects, it is to indicate whether the underlying :class:`TensorValued` object where the
        unary operation is performed onto, is valid or not.
        """
        return self.ref.is_valid

    @property
    def raw_data(self):
        """The data of the underlying :class:`TensorValued` object where the unary operation is performed onto."""
        return self.ref.contract()

    def fix_index(self, index, fix_to=0):
        """Fix the given index to the given value. The object after the method would have the same type as the original
        one, with rank 1 smaller than the original.

        :param index: The index to fix.
        :type index: :class:`int`.
        :param fix_to: the value to assign to the given index.
        :type fix_to: :class:`int`.

        :returns:  :class:`TensorView` -- The :class:`TensorView` object after fixing the given index.
        """
        self.ref = self.ref.fix_index(index, fix_to)

    def expand(self, recursive=False):
        """Commute the unary operation with the underlying tensor network, when the unary operation is a homomorphism
        for tensor network contractions."""
        from acqdp.tensor_network import TensorNetwork
        if not self.homomorphism or not isinstance(self.ref, TensorNetwork):
            return self
        else:
            k = self.ref.copy()
            if recursive:
                k.expand(recursive=True)
        for node_name in k.nodes_by_name:
            k.update_node(node_name, TensorView(k.network.nodes[(0, node_name)]['tensor'], self.func, self.homomorphism))
        return k

    def cast(self, dtype):
        self.dtype = dtype
        self.ref = self.ref.cast(dtype)
        return self

    def contract(self, **kwargs):
        return self.func(self.ref.contract(**kwargs))

    def copy(self):
        return TensorView(self.ref.copy(), self.func)

    def __deepcopy__(self, memo):
        return TensorView(copy.deepcopy(self.ref), self.func)
