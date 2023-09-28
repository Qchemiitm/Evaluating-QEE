# Deciphering the Potential of Qubit Efficient Encoding Algorithm for Multiple Applications

## Description
This project explores the potential of qubit-efficient encoding (QEE) scheme for simulating molecular systems by 
- Estimating the ground state energy values for several molecules such as $H_2$, $LiH$, $BeH_2$ and $H_2O$ and comparing these values against the Jordan-Wigner and Parity encoded Hamiltonian as well as classical values. 
- Evaluating the excited state energy values for a couple of molecules
- Identifying the geometrical configuration corresponding to minimum energy values
- Evaluating the dipole moments for a few molecules

## Pre-requisites
The pre-requisites for installing the package are:

### Python==3.8.13
It is advisable to create a new environment using either pip or conda to deploy the project. 
If using conda, the following command can be used where \<envname> needs to be replaced with the appropriate name during execution. 
    
    conda create --name <envname> python==3.8.13 

### Qiskit packages
- qiskit==0.38.0
    - qiskit-aer==0.11.0
    - qiskit-ibm-experiment==0.2.6
    - qiskit-ibmq-provider==0.19.2
    - qiskit-terra==0.21.2
- qiskit-nature==0.4.5
- qiskit_experiments==0.4.0

Following commands can be used to install the qiskit packages.

    pip install qiskit==0.38.0
    pip install qiskit-nature==0.4.5
    pip install qiskit_experiments==0.4.0

### PySCF library
- pyscf==2.1.1

Following command can be used to install the PySCF library.

    pip install pyscf==2.1.1

### YAML library
- PyYAML==6.0

Following command can be used to install the PyYAML library.

    pip install PyYAML==6.0

### Matplotlib library
- matplotlib==3.6.0
- pylatexenc==2.10

Following commands can be used to install the Matplotlib library.

    pip install matplotlib==3.6.0
    pip install pylatexenc==2.10 

> Note: It is important that the qiskit account credentials are stored on the machine on which the code is installed and executed. Refer to IBM Qiskit help to identify how qiskit account credentials can be stored locally.

## Usage

### Estimating ground state and excited state energies

Run the program *run_energy_estimator.py* at the command prompt using the command

    python3 run_energy_estimator.py

The above program will execute for the molecule configured in the parameters.yaml file. Also, it will print an output at the end of the execution. A sample output for $H_2$ molecule is as follows.

    ====================================================================
    The minimum ground state value for molecule H2 having bond length 0.7000000000000001 using mapper QEE is: -1.1361894476007042
    The ground state value for molecule H2 having bond length 0.7000000000000001 using mapper JW is: -1.1173407654981196
    The ground state value for molecule H2 having bond length 0.7000000000000001 using mapper JW and solved via exact eigen solver is: -1.1361894540659203
    The ground state value for molecule H2 having bond length 0.7000000000000001 using mapper P is: -1.1361894522376725
    The ground state value for molecule H2 having bond length 0.7000000000000001 using mapper P and solved via exact eigen solver is: -1.1361894540659208
    The 1st excited state value for molecule H2 having bond length 0.7000000000000001 using mapper QEE is: -0.478471220474864
    The 1st excited state value for molecule H2 having bond length 0.7000000000000001 using mapper JW is: -0.5017753769578182
    The 1st excited state value for molecule H2 having bond length 0.7000000000000001 using mapper P is: -0.47846433248820697
    The 2nd excited state value for molecule H2 having bond length 0.7000000000000001 using mapper QEE is: -0.12047006634444113
    The 2nd excited state value for molecule H2 having bond length 0.7000000000000001 using mapper JW is: -0.1359670119853097
    The 2nd excited state value for molecule H2 having bond length 0.7000000000000001 using mapper P is: -0.12046318023960856
    ====================================================================

Additionally, plots comparing the QEE, JW and Parity outputs along with the exact solution via minimum eigensolver approach will be created in the ./output folder. Furthermore, the ansatz circuit used for all the $3$ encoding schemes will be created in the same ./output folder. Sample Ansatz circuits for $H_2$ molecule, the first one using Jordan-Wigner encoding and the second using QEE encoding are shown below.

![ansatz_JW_H2](https://github.com/Qchemiitm/Evaluating-QEE/assets/119607439/045edf0c-a646-4597-95fe-7d3a5d498e37)

![ansatz_QEE_H2](https://github.com/Qchemiitm/Evaluating-QEE/assets/119607439/9fb9eb2a-867e-4726-b57f-cf7df2597a2f)

There are several parameters that can be tweaked to execute different scenarios. The parameters are provided in the parameters.yaml file. Some of the parameters are:

    # Allowed molecule values: Enter one of "H2", "BeH2", "C", "LiH", "H2O"
    # Allowed optimizer_label values: COBYLA, L-BFGS-B, SLSQP, SPSA.
    # Allowed entanglement_type values: linear (default), full
    # Allowed rotations values: "ry", ["ry","rz"]
    # Allowed run_es_solver: 1 (choose 1 to execute qEOM), 0 (-)
    # Allowed is_solve_exact: 1 (choose 1 to solve using NumpyMinimumEigenSolve), 0(-)
    # Allowed enable_debug values: 0, 1 (choose 1 to print debug logs)  
    # Allowed distance values: list representing [start,end,step] or individual values e.g. [[1],[2]]

Additionally, the following parameters can also be edited:

    depth: 2 // number of layers or depth of the circuit
    optimizer_maxiter: 300 // maximum number of iterations for the optimizer

> Note: Executing the program will create an *output* and *logs* folders within the current directory.
> Note: Water ($H_2O$) molecule has non-linear geometry and hence exhibits different behaviour as compared to the other molecules considered for evaluation. Hence, one can execute the program h2o_run_energy_estimator.py using the command

    python3 h2o_run_energy_estimator.py


### Minimum energy configuration search 

There are $2$ separate programs: *run_min_conf_search_h2.py* and *run_min_conf_search_lih.py*. Each of the programs runs in $2$ phases:

1. Estimate the ground state energy for different interatomic distance values using VQE algorithm and QEE encoding scheme.
2. Run gradient descent with an appropriate learning parameter and convergence criteria to identify the bond length for the minimum energy value.

Run any of the above programs at the command prompt using the command

    python3 run_min_conf_search_h2.py


### Evaluating dipole moments 

Run the program *compute_dipole_moment.py* to compute the dipole moment of the molecule set in the parameters.yaml file. One can use the command

    python3 compute_dipole_moment.py


## References

1. The QEE code (https://github.com/m24639297/qubit-efficient-mapping) has been updated to make it compatible with the later versions of qiskit-nature, namely 0.4.5.


## Author(s)

Amit Shashikant Kesari


## License
[Apache2.0](https://opensource.org/licenses/Apache-2.0)
