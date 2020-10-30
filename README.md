# Alibaba Cloud Quantum Development Platform (ACQDP)

## Introduction
ACQDP is an open-source simulator-driven development tool for quantum algorithms and quantum computers. The initial release of ACQDP in October 2020 features Alibaba Quantum Laboratory’s general-purpose, tensor-contraction based  quantum circuit simulator, together with some applications on quantum algorithm and error correction simulations. Some future directions of ACQDP of higher prioritites are

1. Strengthening the capabilities of the simulator, in terms of the scale of the target circuits, and allowing approximations.
2. Improving the capabilities for and expanding the scope of applications.
3. Developing friendly user interfaces for both the educational and research communities.
4. Adding utilities facilitating the deployment in various computing environments.

## Computing Engine
Partially inspired by the recent quantum supremacy experiment, classical simulation of quantum circuits attracts quite a bit of attention and impressive progress has been made along this line of research to significantly improve the performance of classical simulation of quantum circuits. Key ingredients include
1. Quantum circuit simulation as tensor network contraction [[1]](#1);
2. Undirected graph model formalism[[2]](#2);
3. Dynamic slicing [[3]](#3);
4. Contraction tree [[4]](#4);
5. Contraction subtree reconfiguration [[5]](#5).

We are happy to be part of this effort.

## Use Cases

* Efficient exact contraction of intermediate-sized tensor networks
* Deployment on large-scale clusters for contracting complex tensor networks
* Efficient exact simulation of intermediate sized quantum circuit
* Classical simulation under different quantum noise models

## Documentation
[See full documentation here.](https://alibabaquantumlab.github.io/acqdp)

## Installation
[See instructions here.](https://alibabaquantumlab.github.io/acqdp/installation.html)

## Contributing

If you are interested in contributing to ACQDP feel free to contact me or create an issue on the issue tracking system.

## References

<a id="1">[1]</a>
Markov, I. and Shi, Y.(2008)
Simulating quantum computation by contracting tensor networks
SIAM Journal on Computing, 38(3):963-981, 2008

<a id="2">[2]</a>
Boixo, S., Isakov, S., Smelyanskiy, V. and Neven, H. (2017)
Simulation of low-depth quantum circuits as complex undirected graphical models
arXiv preprint arXiv:1712.05384

<a id="3">[3]</a>
Chen, J., Zhang, F., Huang, C., Newman, M. and Shi, Y.(2018)
Classical simulation of intermediate-size quantum circuits
arXiv preprint arXiv:1805.01450

<a id="4">[4]</a>
Zhang, F., Huang, C., Newman M., Cai, J., Yu, H., Tian, Z., Yuan, B., Xu, H.,Wu, J., Gao, X., Chen, J., Szegedy, M. and Shi, Y.(2019)
Alibaba Cloud Quantum Development Platform: Large-Scale Classical Simulation of Quantum Circuits
arXiv preprint arXiv:1907.11217

<a id="5">[5]</a>
Gray, J. and Kourtis, S.(2020)
Hyper-optimized tensor network contraction
arXiv preprint arXiv:2002.01935

<a id="6">[6]</a>
Huang, C., Zhang, F.,Newman M., Cai, J., Gao, X., Tian, Z., Wu, J., Xu, H., Yu, H., Yuan, B.,\
 Szegedy, M., Shi, Y. and Chen, J. (2020)
Classical Simulation of Quantum Supremacy Circuits
arXiv preprint arXiv:2005.06787
