import numpy
import copy
from collections import OrderedDict
from .tensor_valued import TensorValued, DTYPE


class TensorSum(TensorValued):
    """A :class:`TensorSum` object represents the summation of multiple tensors.

    :ivar terms_by_name: a dictionary with key-value pairs, where the key is the name of a summand and the value is the
        corresponding summand :class:`TensorValued` object.
    """

    def __init__(self, terms=None, dtype: type = DTYPE) -> None:
        """The constructor of a `TensorSum` object."""
        super().__init__(dtype)
        if terms is None:
            self.terms_by_name = OrderedDict()
        else:
            for term_name in self.terms_by_name:
                self.terms_by_name[term_name] = terms[term_name]

    def __str__(self):
        term_str = "\nTerms:"
        for term_name in self.terms_by_name:
            term_str += "\n" + str(term_name) + "\n" + str(self.terms_by_name[term_name])
        return super().__str__() + term_str

    def _update_shape(self, curr, tmp):
        if tmp is None:
            return curr
        if curr is None:
            return list(tmp)
        if len(curr) != len(tmp):
            raise ValueError('Component shapes do not match')
        for i in range(len(curr)):
            if curr[i] is None:
                curr[i] = tmp[i]
            elif (tmp[i] is not None) and (tmp[i] != curr[i]):
                raise ValueError('Component shapes do not match')
        return curr

    def _invalidate_shape_cache(self):
        if hasattr(self, '_cached_shape'):
            del self._cached_shape

    @property
    def shape(self):
        """The common property of all :class:`TensorValued` classes, yielding the shape of the object.

        :class:`TensorValued` objects must have compatible shapes in order to be connected together in a
        :class:`TensorNetwork`, or summed over in a :class:`TensorSum`.
        """
        if not hasattr(self, '_cached_shape'):
            curr = None
            for tsr_name in self.terms_by_name:
                tsr = self.terms_by_name[tsr_name]
                curr = self._update_shape(curr, tsr.shape)
            self._cached_shape = curr
        return tuple(self._cached_shape) if self._cached_shape is not None else None

    @property
    def is_valid(self):
        """The common property of all :class:`TensorValued` classes, indicating whether the :class:`TensorValued` object
        is valid or not.

        In every step of a program, all existing :class:`TensorValued` object must be valid, otherwise an exception
        should be thrown out; this property is for double checking that the current :class:`TensorValued` object is
        indeed valid.
        """
        try:
            self.shape
        except ValueError:
            return False
        else:
            return True

    @property
    def is_ready(self):
        """The common property of all :class:`TensorValued` classes, indicating whether the current
        :class:`TensorValued` object is ready for contraction, i.e. whether it semantically represents a tensor with a
        definite value.

        In the process of a program, not all :class:`TensorValued` objects need to be ready; however once the `data`
        property of a certain object is queried, such object must be ready in order to successfully yield an
        :class:`numpy.ndarray` object.
        """
        for t in self.terms_by_name.values():
            if not t.is_ready:
                return False
        return self.is_valid

    def add_term(self, term=None, tensor=None):
        """Add a term to the summation.

        :param term: Name of the term to be added. If not given, an auto-assigned one will be given as the output.
        :type term: hashable

        :param tensor: Value of the term to be added.
        :type tensor: :class:`TensorValued` or None

        :returns: The name of the newly added term.
        """
        from .tensor import Tensor
        if not isinstance(tensor, TensorValued):
            tensor = Tensor(tensor)
        if term is None:
            term = tensor.identifier
        if tensor.dtype == complex:
            self.dtype = complex
        if term in self.terms_by_name:
            raise KeyError("term {} to be added into the tensor network already in the tensor network!".format(term))
        if tensor.shape is not None:
            self.shape  # Make sure the shape cache is initialized
            self._cached_shape = self._update_shape(self._cached_shape, tensor.shape)
        self.terms_by_name[term] = tensor
        return term

    def __iadd__(self, t):
        self.add_term(tensor=t)
        return self

    def update_term(self, term, tensor=None):
        """Update the value of a term in the summation.

        :param term: Name of the term to be updated.
        :type term: hashable
        :param tensor: New value of the term
        :type tensor: :class:`TensorValued`

        :returns: Name of the term to be updated.
        """
        from .tensor import Tensor
        if (type(tensor) == numpy.ndarray) or (tensor is None):
            tensor = Tensor(tensor)
        if term not in self.terms_by_name:
            raise KeyError("term {} not in the TensorSum object".format(term))
        self.terms_by_name[term] = tensor
        self._invalidate_shape_cache()
        return term

    def remove_term(self, term):
        """Remove a term from the summation.

        :param term: Name of the term to be removed.
        :type term: hashable
        :returns: :class:`TensorValued` Value of the removed term
        """
        pop = self.terms_by_name.pop(term)
        self._invalidate_shape_cache()
        return pop

    def fix_index(self, index, fix_to=0):
        """Fix the given index to the given value. The result :class:`TensorValued` object would have the same type as
        the original one, with rank 1 smaller than the original.

        :param index: The index to fix.
        :type index: :class:`int`.
        :param fix_to: The value to assign to the given index.
        :type fix_to: :class:`int`.
        :returns:  :class:`TensorValued` -- The :class:`TensorValued` object after fixing the given index.
        :raises: NotImplementedError
        """
        ts = self.copy()
        for term in ts.terms_by_name:
            ts.terms_by_name[term] = ts.terms_by_name[term].fix_index(index, fix_to)
        ts._invalidate_shape_cache()
        return ts

    def cast(self, dtype):
        self.dtype = dtype
        for term in self.terms_by_name:
            self.update_term(term, self.terms_by_name[term].cast(dtype))
        return self

    def contract(self, **kwargs):
        """Evaluate the object by summing over all the terms.

        :returns: :class:`numpy.ndarray`
        """
        res = [
            self.terms_by_name[term].contract(**kwargs)
            for term in self.terms_by_name
        ]
        return sum(res)

    def copy(self):
        ts = TensorSum(dtype=self.dtype)
        for t in self.terms_by_name:
            ts.add_term(t, self.terms_by_name[t])
        return ts

    def __deepcopy__(self, memo):
        ts = TensorSum(dtype=self.dtype)
        for t in self.terms_by_name:
            ts.add_term(t, copy.deepcopy(self.terms_by_name[t]))
        return ts
