# FedSysID_Python

Python 3.10.13 implementation of **FedSYS**, a federated learning algorithm for system identification. This is a direct translation of the original MATLAB code by Han Wang et al. (2022).

In addition to FedSYS, **FedProx** is also implemented to compare **FedLin**, **FedProx**, and **FedAvg**. A script to benchmark the execution time of these algorithms is included.

For more details, check the original papers:  
- **FedSYS**: [A Federated Learning Approach for System Identification](https://arxiv.org/abs/2211.14393) by Han Wang et al. (2022)  
- **FedProx**: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) by Tian Li et al. (2020)

## Requirements

To run the code, it is advised to create a new virtual Python environment and install Jupyter.

Then, the required libraries are listed in the `requirements.txt` file. To install them, simply navigate to the corresponding folder and run the following command:

```bash
pip install -r requirements.txt