"""
This module contains classes and methods for setting the molecule driver and executing the Ground state solver or QEOM using VQE algorithm
Author(s): Amit S. Kesari
"""
from fileinput import close
from qiskit import transpile, Aer, IBMQ
import matplotlib.pyplot as plt
import numpy as np
from qiskit_nature.drivers import Molecule, UnitsType
from qiskit_nature.drivers.second_quantization import \
            ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper

from qiskit_nature.settings import settings
settings.dict_aux_operators = True

from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.circuit.library import HartreeFock
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit_nature.circuit.library.ansatzes import UCC
from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA, SLSQP, SPSA

from qee import groupedFermionicOperator as groupFOp
from VQECustom import VQENoAuxOperators
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

class EnergyEstimator:

    """
    Key concepts: The key concepts important for this class are:

    1. VQE: It stands for Variational Quantum EigenSolver. It essentially is a hybrid algorithm 
        that executes a quantum circuit on quantum simulator/device, but the parameters of the QC
        are optimized using a classical optimizer. Here, we will use the in-built VQE algorithm 
        available in Qiskit library.

    2. Ansatz: It is the initial parameterized QC that acts as the starting point. It is also known as 
        the variational form. Based on evolution of the parameters through the classical optimizer, 
        the QC also evolves.
        There are 2 types of variational forms:
        1. Domain specific: e.g. UCSSD (better when more qubits are available)
        2. General approach: Gates are layered such that good approximations on wide range of states are
            obtained e.g. TwoLocal class having RyRz, Ry and SwapRz as rotation blocks, 
            CX, CZ, etc. as entanglement blocks. The depth setting (d = reps) indicates how many times the 
            variational form has to be repeated.
        
        We will be using the TwoLocal class for heuristic based Ansatz.
        https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html.
        
        class TwoLocal(num_qubits=None, rotation_blocks=None, entanglement_blocks=None, 
                        entanglement='full', reps=3, skip_unentangled_qubits=False, 
                        skip_final_rotation_layer=False, parameter_prefix='θ', insert_barriers=False, 
                        initial_state=None, name='TwoLocal')
        e.g. ansatz = TwoLocal(num_qubits=num_spin_orbitals, rotation_blocks=[ry,rz], 
                                entanglement_blocks = [cz], entanglement='linear',
                                insert_barriers=True)

    3. Initial state: This is the initial state of the qubits that can be passed to an Ansatz. Usually, 
        the initial state would be a zero state or Hartree-Fock state depending on the mapper/transformation being used i.e. Jordan-wigner vs. QEE.

    4. Classical Optimizer: There are several classical optimizers provided as part of the Qiskit 
        algorithms library.
        https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.html

        We have tried the following options
        1. L_BFGS_B (Limited memory BFGS bound optimizer) 
        2. COBYLA (Constrained optimization by linear approximation optimizer)
        3. Simultaneous Perturbation Stochastic Approximation optimizer (SPSA) - recommended for noisy
            devices and simulators

    5. Driver: A quantum chemistry problem needs to be mapped to a driver. 
                We use the PySCF driver (Python-based Simulations of Chemistry Framework). 
                It is essentially, a classical code for the second-quantization driver.
                https://pyscf.org/
        
        class ElectronicStructureMoleculeDriver(molecule, basis='sto3g', method=MethodType.RHF, 
                                                driver_type=ElectronicStructureDriverType.AUTO, 
                                                driver_kwargs=None)                            
        e.g. driver = ElectronicStructureMoleculeDriver(
                        in_molecule, basis='sto3g', driver_type=ElectronicStructureDriverType.PYSCF)                             

        Transformers help to modify the required values of the drivers and update them back to the input.
        e.g. freeze_core=True,remove_orbitals=[3,4]

    6. Mapping electronic structure problems to qubits: For fermionic problems, 3 types of mappers 
        are available:
        1. Jordan Wigner
        2. Parity
        3. Bravyi Kitaev
        4. Qubit-efficient encoding (QEE)

        We have used QEE as the default mapper along with two-qubit-reduction as it helps in 
        reducing the number of required qubits when mapping the 2nd quantized hamiltonian.
    """    
    
    # define constructor
    def __init__(self, geometry, multiplicity=1, freeze_core=False, orbital_reduction_list=None, charge=0, unit=UnitsType.ANGSTROM, basis='sto3g', debug=False) -> None:
        """
        This method initializes the molecule, sets the driver, applies the necessary transformation
        and then finally defines the electronic structure problem.
        Input arguments: 
        1. geometry e.g. geometry = [["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.57]]]
        2. z2symmetry_reduction e.g. [-1,1]
        3. orbital_reduction_list e.g. [3,4]
        4. mapper e.g. P (for Parity mapper - default), JW (for Jordan-Wigner)
        5. charge e.g. 0 (default)
        6. unit e.g. UnitsType.ANGSTROM - applicable to distances in geometry argument
        7. basis e.g. 'sto3g' (default as it is the minimum basis)
        8. debug e.g. True or False
        """
        log.info(f"Inside constructor of EnergyEstimator class.")

        # initialize the molecule structure
        self.molecule = Molecule(geometry=geometry, multiplicity=multiplicity, charge=charge, units=unit)
        # initialize the driver
        self.driver = ElectronicStructureMoleculeDriver(
                            self.molecule, basis=basis, 
                            driver_type=ElectronicStructureDriverType.PYSCF)
               
        # Apply transformations to the driver to get the electronic structure problem, if applicable
        if freeze_core == True and orbital_reduction_list is not None:
            molecule_transformers = \
                [FreezeCoreTransformer(freeze_core=True,remove_orbitals=orbital_reduction_list)]
            self.electronic_structure_problem = \
                ElectronicStructureProblem(self.driver, molecule_transformers)
        elif freeze_core == True:
            molecule_transformers = \
                [FreezeCoreTransformer(freeze_core=True)]
            self.electronic_structure_problem = \
                ElectronicStructureProblem(self.driver, molecule_transformers)
        else:
            self.electronic_structure_problem = \
                ElectronicStructureProblem(self.driver)
            
        self.second_quantized_op = self.electronic_structure_problem.second_q_ops()

        if debug == True:
            log.debug(f"Printing molecule structure ...")
            log.debug(self.molecule)
            log.debug(f"Printing electronic structure problem ...")
            log.debug(vars(self.electronic_structure_problem))
            log.debug(f"Printing second quantized hamiltonian ...")
            log.debug(self.second_quantized_op)
            
        log.info(f"End of constructor of EnergyEstimator class.")

    def set_mapper(self, mapper, z2symmetry_reduction=None, hf_mode = 'rhf', qee_max_allowed_processes=8, debug=False):
        """
        Description: This method assigns the appropriate mapper/transformation to the electronic hamiltonian
        """
        log.info(f"Setting mapper/transformation for mapper {mapper} and z2 symmetry reduction as  {z2symmetry_reduction} ...")
        # initialize the converter
        try:
            if mapper == 'P': # P indicates parity mapper
                if z2symmetry_reduction is not None and bool(z2symmetry_reduction) == True:
                    self.qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True, 
                                            z2symmetry_reduction=z2symmetry_reduction)
                else:
                    self.qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
            elif mapper == 'JW': # JW indicates Jordan-Wigner mapper
                if z2symmetry_reduction is not None and bool(z2symmetry_reduction) == True:
                    self.qubit_converter = QubitConverter(mapper=JordanWignerMapper(),  
                                            two_qubit_reduction=True, 
                                            z2symmetry_reduction=z2symmetry_reduction)
                else:
                    self.qubit_converter = QubitConverter(mapper=JordanWignerMapper())      
            elif mapper == 'QEE': # Qubit efficient encoding
                alpha_particles, beta_particles = self.electronic_structure_problem.num_particles
                gQEE = groupFOp.groupedFermionicOperator(
                             #fermionic_hamiltonian=self.second_quantized_op,
                             num_spin_orbitals=self.electronic_structure_problem.num_spin_orbitals,
                             num_electrons=alpha_particles + beta_particles,
                             mode=hf_mode,
                             max_allowed_processes=qee_max_allowed_processes,
                            ) 
                self.qubit_converter = QubitConverter(mapper=gQEE)
            else:
                raise ValueError("Incorrect input for the mapper: " + mapper)
        except ValueError as ve:
            log.exception(ve, stack_info=True)
            raise

        log.info(f"Mapper/transformation {mapper} set.")

    def get_hamiltonian_op(self, debug=False):
        """
        Description: This method returns the qubit equivalent to the second quantized hamiltonian
        i.e. Electronic Energy
        Input arguments:
        1. Debug e.g. True/False
        """
  
        log.info(f"Inside get_hamiltonian_op method.")
    
        self.qubit_hamiltonian_op = self.qubit_converter.convert(
                                        self.second_quantized_op["ElectronicEnergy"], 
                                        num_particles=self.electronic_structure_problem.num_particles)

        if debug == True:
            log.debug("Printing Qubit converted hamiltonian ...")
            log.debug(self.qubit_hamiltonian_op)
        
        log.info(f"Qubit converted hamiltonian set.")
            
        return self.qubit_hamiltonian_op, self.qubit_converter.z2symmetries
        
    def set_initial_state(self, debug=False):
        """
        Description: This method sets the initial state i.e. Hatree-Fock state
        Input arguments:
        1. filename_with_path e.g. file location where the circuit diagram will be saved
        2. debug e.g. True or False
        """
        
        log.info(f"Inside set_initial_state method.")

        # set the initial Hartree Fock state
        initial_state = \
            HartreeFock(num_spin_orbitals=self.electronic_structure_problem.num_spin_orbitals, 
                        num_particles=self.electronic_structure_problem.num_particles, 
                        qubit_converter=self.qubit_converter
                       )
        
        if debug == True:
            log.debug(f"Printing initial state ...")
            log.debug(initial_state)
        
        log.info(f"Initial state set.")
        return initial_state

    # defining optimizer as static method as it is not object dependent
    @staticmethod
    def get_optimizer(optimizer_label, maxiter=500, debug=False):
        """
        Description: This method identifies the classical optimizer based on input i.e. L_BFGS_B, 
        COBYLA or SPSA. SPSA is recommended for noisy simulators and quantum devices.
        Input arguments:
        1. Optimizer_label e.g. L-BFGS-B -> return L_BFGS_B(), COBYLA, SPSA
        2. debug e.g. True or False
        """
        
        log.info(f"Inside get_optimizer method.")

        # get the required optimizer
        try:
            if optimizer_label == 'L-BFGS-B':
                optimizer = L_BFGS_B(maxiter=maxiter)
            elif optimizer_label == 'COBYLA':
                optimizer = COBYLA(maxiter=maxiter)
            elif optimizer_label == 'SLSQP':
                optimizer = SLSQP(maxiter=maxiter)
            elif optimizer_label == 'SPSA':
                optimizer = SPSA(maxiter=maxiter)
            else:
                raise ValueError(f"Optimizer label values can only be one of L_BFGS_B, COBYLA or SLSQP: {optimizer_label}")
        except ValueError as ve:
            log.exception(ve, stack_info=True)
            raise
        else:
            pass
        
        if debug == True:
            log.debug(f"Printing optimizer ...")
            log.debug(optimizer)
        
        log.info(f"Optimizer set.")
        return(optimizer)

    def build_ansatz(self, num_qubits, initial_state, rotations, entanglement, 
                     entanglement_type='linear', depth=2, debug=False):
        """
        Description: This method uses the TwoClass module to build the Ansatz based on initial state, 
        rotation blocls, entanglement blocls, entanglement type and depth of the circuit.
        Input arguments:
        1. num_qubits: Number of qubits
        2. initial_state:
        3. rotations:
        4. entanglement:
        5. entanglement_type: linear (only adjacent qubits are interconnected) or full (all qubits are
            interconnected)
        """
        
        log.info(f"Inside build_ansatz method.")

        # define the ansatz based on TwoLocal
        ansatz = TwoLocal(num_qubits=num_qubits,
                            rotation_blocks=rotations, entanglement_blocks=entanglement,
                            entanglement=entanglement_type, reps=depth,
                            insert_barriers=True
                         )
    
        # add the initial state to ansatz, if applicable
        if initial_state is not None:
            ansatz.compose(initial_state, front=True, inplace=True)

        if debug == True:
            log.debug(f"Printing ansatz ...")
            log.debug(ansatz)
            log.debug(f"End of build_ansatz method.")
        
        log.info(f"Ansatz set.")
        return(ansatz)

    def build_ucc_ansatz(self, initial_state, depth=2, debug=False):
        """
        Description: This method uses the UCC module (specifically UCCSD) to build the Ansatz.
        Input arguments:
        1. initial_state:
        2. depth/reps of the ansatz circuit
        """
        
        log.info(f"Inside build_ucc_ansatz method.")

        # define the ansatz based on UCCSD
        excitations = 'sd'
        ansatz = UCC(excitations=excitations,
                    num_particles=self.electronic_structure_problem.num_particles,
                    num_spin_orbitals=self.electronic_structure_problem.num_spin_orbitals,
                    #initial_state=initial_state,
                    qubit_converter=self.qubit_converter,
                    reps=depth
                )
    
        # add the initial state to ansatz, if applicable
        if initial_state is not None:
            ansatz.compose(initial_state, front=True, inplace=True)

        if debug == True:
            log.debug(f"Printing UCC ansatz ...")
            log.debug(ansatz)
            log.debug(f"End of build_ucc_ansatz method.")
        
        log.info(f"UCCSD Ansatz set.")
        return(ansatz)

def exact_eigen_solver(hamiltonian_op):
    """
    Description: This method is helpful for small molecules wherein the enrgy values can be 
    solved in an exact manner.
    Input arguments:
    1. hamiltonian_op
    """
    npme = NumPyMinimumEigensolver()
    exact_result = npme.compute_minimum_eigenvalue(operator=hamiltonian_op)
    reference_value = exact_result.eigenvalue.real

    return(reference_value)

def variational_eigen_solver(ansatz, optimizer, quantum_instance, is_support_aux_operators = True, use_callback='N'):
    """
    Description: This method calls the in-built VQE algorithm available as part of Qiskit library
    As part of enhancement, a custom eigen solver can also be implemented.
    Input arguments:
    1. ansatz
    2. optimizer
    3. quantum_instance (based on backend and nosie model, if applied)
    4. Using callback if set
    """
    log.info(f"Invoking variational eigen solver with support for auxiliary operators = {is_support_aux_operators} and using callback functionality = {use_callback}")

    if is_support_aux_operators == False:
        if use_callback == 'Y':
            vqe_output = VQENoAuxOperators(ansatz,optimizer=optimizer,quantum_instance=quantum_instance,
                                callback = store_intermediate_results)
        else:
            vqe_output = VQENoAuxOperators(ansatz,optimizer=optimizer,quantum_instance=quantum_instance)
    else:
        if use_callback == 'Y':
            vqe_output = VQE(ansatz,optimizer=optimizer,quantum_instance=quantum_instance,
                                callback = store_intermediate_results)
        else:
            vqe_output = VQE(ansatz,optimizer=optimizer,quantum_instance=quantum_instance)
    return(vqe_output)

# this is a callback function to store intermediate results for VQE
intermediate_eval_count = []
intermediate_mean = []
intermediate_params = []
intermediate_std = []
def store_intermediate_results(eval_count, parameters, mean, std):
    intermediate_eval_count.append(eval_count)
    intermediate_mean.append(mean)
    intermediate_params.append(parameters)
    intermediate_std.append(std)

def plot_energy_graph(distances, energies, debug=False):
    """
    Description: Plot the energy vs inter-atomic distance graph
    """ 
    log.info(f"Inside plot_energy_graph method.")

    energy_plot = multiple_plot_graph(x_values=distances, y_values=energies, 
                      plot_title='Energy vs. Interatomic Distance', x_label='Distance (Angstrom)', 
                      y_label='Energy (hartree)', 
                      debug=debug)
    
    log.info(f"End of plot_energy_graph method.")

    return(energy_plot)

def plot_dipole_moment_graph(distances, dipole_values, debug=False):
    """
    Description: Plot the dipole moment vs inter-atomic distance graph
    """ 
    log.info(f"Inside plot_dipole_moment_graph method.")

    dipole_moment_plot = multiple_plot_graph(x_values=distances, y_values=dipole_values, 
                            plot_title="Dipole Moment vs. interatomic distance", x_label="Distance (Angstrom)", y_label="Dipole Moment (a.u)",
                            debug=debug)
    
    log.info(f"End of plot_dipole_moment_graph method.")

    return(dipole_moment_plot)

def plot_intermediate_results_graph(debug=False):
    """
    Description: Plot the count vs mean values of intermediate results
    """
    log.info(f"Inside plot_intermediate_results_graph method.")

    intermediate_results_plot = \
        single_plot_graph(x_values=intermediate_eval_count, y_values=intermediate_mean, 
                          plot_title='Energy vs. Evaluation count', 
                          x_label='Evaluation Count', y_label='Energy (hartree)',
                          debug=debug)


    log.info(f"End of plot_intermediate_results_graph method.")

    return(intermediate_results_plot)

def single_plot_graph(x_values, y_values, plot_title, x_label, y_label, is_annotate=False, debug=False):
    
    log.info(f"Inside single_plot_graph method.")
    if debug == True:
        log.debug(f"Printing X and Y values ...")
        log.debug(x_values)
        log.debug(y_values)

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)

    #plt.plot(x_values, y_values)
    #plt.xlabel(x_label)
    #plt.ylabel(y_label)
    #plt.title(plot_title)
    #plt.yticks(np.arange(np.floor(min(y_values)), np.ceil(max(y_values)), 0.1))

    if is_annotate == True:
        annot_min(x_values, y_values)

    #plt.legend()
    #plt.show()
    #plt.close()

    log.info(f"End of single_plot_graph method.")
    
    return(plt)

def multiple_plot_graph(x_values, y_values, plot_title, x_label, y_label, debug=False):
    
    log.info(f"Inside multiple_plot_graph method.")
    if debug == True:
        log.debug(f"Printing X and Y values ...")
        log.debug(x_values)
        log.debug(y_values)

    fig, ax = plt.subplots()
    for key in y_values:
        ax.plot(x_values, y_values[key], label=key)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.legend(loc='upper center')

    #plt.show()
    #plt.close()


    log.info(f"End of multiple_plot_graph method.")
    
    return(plt)

# method to display min value on graph
def annot_min(x,y, ax=None):
    ymin = min(y)
    xmin = x[y.index(ymin)]
    text= "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.96), **kw)