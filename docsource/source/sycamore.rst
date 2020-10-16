Demo: Simulating the Sycamore random quantum circuits
===========================================================

The concept of quantum supremacy is first introduced by John Preskill in [P12]_, indicating a milestone that a quantum computer can achieve a certain task that is infeasible for even the most powerful classical computational resource. Quantum supremacy does not immediately indicate the usefulness of quantum computers on solving classically intractable computational problems; it serves rather as an early demonstration that quantum computers can potentially outperform classical computers on specific tasks.

After years of rapid development on quantum hardwares, Google claimed to have achieved quantum supremacy by sampling from a family of random circuits using their latest-generation superconducting quantum computer of 53 qubits, called the `Sycamore` quantum computer [AAB+19]_. It is estimated that the quantum computer takes about 200 seconds to sample 1 million bitstrings with a certain quality, while a comparable task would take 10 millenia on the Summit supercomputer. Using the powerful order-finding scheme in the ACQDP, The recent work from Alibaba managed to pin this estimation down to less than 20 days [HZN+20]_. Although it is still far from the 200 seconds from the quantum hardware, this examples serves as a great example of the ACQDP being used for intermediate-sized tensor network contraction, where up to a few thousands of tensors are involved. The source code can be found at `examples.GRCS` and `examples.circuit_simulation`, and the contraction schemes previously found are available at the folder `benchmarks`.


An easy example :math:`m=10`
----------------------------

`examples/GRCS.py` provides a preliminary parser translating `.qsim` files for the Sycamore circuit instances to `acqdp.circuit.Circuit` and then to `acqdp.tensor_network.TensorNetwork`.

To run the Sycamore circuit with 10 layers, run:

.. code-block:: zsh

    python -m examples.circuit_simulation benchmark/circuit_n53_m10_s0_e0_pABCDCDAB.qsim

Without specifying an order, the program finds a contraction order by invoking the Kahypar hypergraph decomposition scheme. In one run of the script, the program outputs:

.. code-block:: python

    Process 0 succeeded with cost = 10.62287901105588, cw = 27.0, num_split = 0

Indicating a contraction order (without slicing) is found, where the number of floating point operations is `10**10.62` and the biggest intermediate tensor has `2**27` entries. No slicing was needed since the tensor sizes are well within the hardware limits of 16 Gigabytes. The program then proceeds with contraction. The total run time for 5 times the contraction on a single laptop reads:

::

    Compute Time       --- 52.8843309879303 seconds ---

A harder example: :math:`m=20`
------------------------------

The previous example demonstrates a full run of the ACQDP on a relatively small instance. It is to be noted that the :math:`m=10` circuits are not the one used in [AAB+19]_ for the quantum supremacy experiment; instead, the full experiment ran the much deeper, :math:`m=20` circuit. Not only simulating the :math:`m=20` quantum circuit is difficult, it takes longer time to find a good contraction order together with index slicing to carry out the simulation efficiently.

Here we choose to include the contraction schemes we found earlier. Note that all Sycamore quantum circuits with the same :math:`m` have identical tensor network structure, and those contraction schemes can be reused for different instances of random circuits. The contraction schemes are stored in the `benchmark` folder.

.. code-block:: zsh

    python -m examples.circuit_simulation benchmark/circuit_n53_m20_s0_e0_pABCDCDAB.qsim -o benchmark/m20_1.json

One get the estimated cost and number of subtasks:

::

    cost = 19.124598309858378, cw = 29.0, num_slice = 25
    Number of subtasks per batch --- 33554432 ---

Performance
------------

In [HZN+20]_ , we compared our theoretical number of floating point operations and projected experimental running time to the existing results:

.. image:: benchmark.pdf
  :width: 700
  :alt: Comparison of FLOPs / projected running time of the simulation tasks


Classical simulation cost and extrapolated running time of sampling from :math:`m`-cycle random circuits with low XEB fidelities.  The dashed lines represent the theoretical number of floating point operations (FLOPs) and the solid lines represent extrapolated running times from the experiments. The two axes are aligned by the theoretical GPU efficiency of an Nvidia V100.Consequently, the dashed lines represent runtime lower bounds provided that GPU efficiency is fully saturated. Numerical data for ACQDP is reported in Table 1. The velvet line is reportedin [AAB+19]_ using the hybrid Schr\"odinger-Feynman algorithm, where the projected running time is estimated from a different architecture than Summit, and so the theretical FLOPs is not shown.

Appendix
************

The Sycamore quantum circuit
-----------------------------

The quantum circuit ran on the Google Sycamore quantum device are drawn from a particular distribution of quantum circuits.

.. image:: circuit.pdf
  :width: 700
  :alt: Sycamore circuit

The structure of the 53-qubit random quantum circuits is shown above. There are 53 qubits on the Sycamore quantum chip, with a pairwise connection graph as shown in (a). The random circuits consists of repetitions of alternations between fixed two-qubit gates followed by random one qubit gates. Lines of different colors in (a) represent two-qubit gates that appear in different layers. (b) shows a schematic diagram of an 8-cycle circuit. Each cycle includes a layer of random single-qubit gates (empty squares in the diagram) and a layer of two-qubit gates (labeled A, B, C, or D, and colored according to the two-qubit gates in (a)). For longer circuits, the layers repeat in the sequence A, B, C, D, C, D, A, B. Note that there is an extra layer of single-qubit gates preceding measurement.

The classical simulation algorithm
----------------------------------

We adopt the tensor network contraction framework proposed in [BIS+18]_, [AAB+19]_ as the basis for our simulation of random circuit sampling. This framework assumes that the outcome distribution of a random quantum circuit is a randomly permuted Porter-Thomas distribution. Under this assumption, we can perform *frugal rejection sampling* on bitstrings by computing the corresponding amplitudes [MFI+18]_. When the batch size of bitstrings is sufficiently large (chosen in our case to be 64), then with high probability, at least one outcome among the batch will be accepted. It can be estimated from [MFI+18]_ that this framework achieves almost perfect sampling when the batch size is chosen to be 64. We can choose the batch to be a state vector on 6 qubits while randomly post-selecting the remaining 47 qubits. In this case, the aggregated result of the amplitudes can be expressed as an open tensor network. This translates the task of sampling from random quantum circuits to the task of contracting a tensor network. For random circuits with :math:`m=12,14,20` cycles, we choose the qubits (0,1,2,3,4,5) in the upper-right corner, and for :math:`m=16, 18` cycles, we choose the qubits (10,17,26,36,27,18) in the lower-right corner. These choices minimize the overhead introduced by simultaneously evaluating each batch of 64 amplitudes.

References
*************************


.. [P12] John Preskill, *Quantum computing and the entanglement frontier*, arXiv preprint arXiv:1203.5813, 2012.
.. [AAB+19] Frank Arute et al, *Quantum supremacy using a programmable superconducting processor*, Nature, 574(7779):505– 510, 2019.
.. [HZN+20] Cupjin Huang, Fang Zhang, Michael Newman, Junjie Cai, Xun Gao, Zhengxiong Tian, Junyin Wu, Haihong Xu, Huanjun Yu, Bo Yuan, Mario Szegedy, Yaoyun Shi, and Jianxin Chen, *Classical Simulation of Quantum Supremacy Circuits*, arXiv preprint arXiv:2005.06787, 2020.
.. [MFI+18] Igor L Markov, Aneeqa Fatima, Sergei V Isakov, and Sergio Boixo.  *Quantum supremacy is both closer and farther than it appears*, arXiv preprint arXiv:1807.10749, 2018.
.. [BIS+18] Sergio Boixo, Sergei V Isakov, Vadim N Smelyanskiy, Ryan Babbush, Nan Ding, Zhang Jiang, Michael J Bremner, John M Martinis, and Hartmut Neven. *Characterizing quantum supremacy in near-term devices*. Nature Physics, 14(6):595–600, 2018.
