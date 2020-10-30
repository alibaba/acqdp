.. _installation:

=================
Overview
=================

ACQDP is an open-source simulator-driven development tool for quantum algorithms and quantum computers. The initial release of ACQDP in October 2020 features Alibaba Quantum Laboratory’s general-purpose, tensor-contraction based  quantum circuit simulator, together with some applications on quantum algorithm and error correction simulations. Some future directions of ACQDP of higher prioritites are

1. Strengthening the capabilities of the simulator, in terms of the scale of the target circuits, and allowing approximations.
2. Improving the capabilities for and expanding the scope of applications.
3. Developing friendly user interfaces for both the educational and research communities.
4. Adding utilities facilitating the deployment in various computing environments.

=================
Installation
=================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

Prerequisites
***************************

`Python <https://docs.python-guide.org/>`__ - version 3.7 or later is required.

`Cython <https://cython.org>`__ (optional) - used in the acqdp to accelerate the software package `KaHyPar <https://github.com/kahypar>`__ used in contraction order finding.
In a command window, run

.. code-block:: bash

   pip install cython

`boost <https://www.boost.org>`__ - C++ boost library, particularly the `program_options <https://www.boost.org/doc/libs/1_58_0/doc/html/program_options.html>`__ library

`KaHyPar python package <https://kahypar.org>`__  (Windows) - On non-Windows systems, it installs automatically
when installing acqdp but it'll error out on Windows. To make it work on Windows, you'll need manually install Kahypar
first following its instructions `here <https://github.com/kahypar/kahypar#the-python-interface>`__. Alternatively, you
can use `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__

Installation from PyPI
**************************

ACQDP packages are published on the `Python Package Index <https://pypi.org/project/ACQDP/>`__ and can be installed using `pip`.
This is the recommended way for most users. However, if you'd like to see or modify the source code, proceed to the next section
`Installation from source code`_

In a command window, run

.. code-block:: bash

   pip install -U acqdp

Installation from source code
*****************************

Alternatively, you can install from source code, say, cloned from `Github <https://github.com/alibaba/acqdp>`__ so you can see and modify the code.
First clone the repo

.. code-block:: bash

   git clone https://github.com/alibaba/acqdp


Then install the ACQDP packages. Enter the `acqdp` folder and run

.. code-block:: bash

   pip install -e .

Example and test codes
***********************

You are ready to go! Find some examples at https://github.com/alibaba/acqdp/tree/master/examples and
https://github.com/alibaba/acqdp/tree/master/demo

To run the example in the examples, for example `GHZ.py`, simply run the following command in the examples folder:

.. code-block:: bash

   python examples/GHZ.py


To see if the package passes all the tests, simply run :command:`python -m pytest`.
