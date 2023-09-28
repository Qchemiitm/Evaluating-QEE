"""
The purpose of this module is to perform minimum configuration search for LiH molecule by estimating the bond length for minimum ground state energy.
Author(s): Amit S. Kesari
"""
## importing basic modules
import os
import numpy as np
## importing qiskit modules
from qiskit import Aer, IBMQ 
# from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
# from qiskit.opflow import StateFn, CircuitStateFn
# from qiskit.compiler import transpile
from scipy.interpolate import CubicSpline

## importing custom modules
from EnergyEstimator import EnergyEstimator, variational_eigen_solver
from QcExecution import QcExecution as qce
from logconfig import get_logger

## define some global variables
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
outputfilepath = curr_dir_path + "/output"

if not os.path.exists(outputfilepath):
    os.makedirs(outputfilepath)

## initialize logger
log = get_logger(__name__)

def is_folder_path_exists(folderpath):
    """
    Check if the folder path location exists and return True, if yes, otherwise False
    """
    ## initialize to False
    folder_exists = False

    try:
        if folderpath is not None:
            #v_file_dir = filename_with_path.rpartition("/")[0]
            try:
                if not os.path.exists(folderpath):
                    raise NameError("Folder path does not exist: " + folderpath)
            except NameError as ne:
                log.exception(ne, stack_info=True)
                raise
            else:
                folder_exists = True
        else:
            raise NameError("Folder path not passed as input: " + folderpath)          
    except NameError as ne1:
        log.exception(ne1, stack_info=True)
        raise
    
    return(folder_exists) 

# setup hydrogen molecule 
def setup_lih_molecule(distance, debug=False) -> EnergyEstimator:
    ## first setup geometry
    geometry = geometry = [["Li", [0, 0, 0]], ["H", [0, 0, distance]]]
    log.info(f"The geometry of molecule LiH is: {geometry}")

    ## setup additional parameters such as freeze_core and remove_orbitals
    multiplicity = 1
    freeze_core = 1
    remove_orbitals = [3, 4]
    ## return energy estimator object
    ee = EnergyEstimator(geometry, multiplicity, freeze_core, remove_orbitals, debug=debug)
    log.info(f"LiH molecule setup for distance {distance}.")
    return (ee)

# identify qubit hamiltonian 
def lih_qubit_hamiltonian(ee, mapper="QEE", debug=False):
    # set QEE mapper
    ee.set_mapper(mapper=mapper, z2symmetry_reduction=None, qee_max_allowed_processes = 8, debug=debug)

    # get hamiltonian operator
    qubit_op, _ = ee.get_hamiltonian_op(debug=debug)

    return qubit_op

# setup VQE and get result
def get_vqe_result(qcex, ansatz, qubit_op, is_simulator=True, debug=False):
    #execute VQE algorithm for the ideal simulator
        
    ## ignore noise model and coupling map as they are not applicable for ideal simulator
    ideal_backend, _, _ = qcex.get_backend(is_simulator=is_simulator, 
                                                    simulator_type='AER_STATEVEVCTOR',
                                                    noise_model_device=None
                                           )
                                                        
    myoptimizer = EnergyEstimator.get_optimizer(optimizer_label='COBYLA', 
                                                maxiter = 300,
                                                debug=debug)

    # execute VQE on ideal/noise-free simulator
    log.info(f"Setting up VQE ... ")
    execute_vqe = variational_eigen_solver(ansatz, optimizer=myoptimizer, 
                                            is_support_aux_operators=False,
                                            quantum_instance=ideal_backend)
    result = execute_vqe.compute_minimum_eigenvalue(qubit_op)
    log.info(f"Hurray! VQE computation is complete.")

    return(result)

# start of main function
def main():
    log.info("=============================================")
    log.info(f"Start of program ...")
    log.info(f"Checking if output path exists ...")
    outputpath_exists = is_folder_path_exists(outputfilepath)
    if outputpath_exists:
        log.info(f"Output path {outputfilepath} present.")

    # load the qiskit account for execution
    log.info(f"Loading Qiskit account. Please wait as this may take a while ...")
    qce.load_account()
    log.info(f"Qiskit account loaded successfully.")

    is_debug = False # setting debug mode to True or False
    is_simulator = 1 ## we only run in simulator mode
    mapper = "QEE" ## we use QEE mapper/transformation for all our computations

    """
    ***************************************************************
    Phase I: In this phase, we compute energy values for different inter-atomic distances
    using VQE.
    ***************************************************************
    """

    # Identify the structure of H2 molecule and get qubit hamiltonian
    molecule = "LiH"
    start_distance = 0.4
    end_distance = 3
    step_size = 0.1
    is_ansatz_printed = False

    distance_list = np.arange(start=start_distance,stop=end_distance,step=step_size)
    total_energy = []

    for dist in distance_list:

        ee = setup_lih_molecule(distance=dist, debug=is_debug)
        qubit_op = lih_qubit_hamiltonian(ee, mapper=mapper, debug=is_debug)

        # setting the appropriate parameter values
        num_particles = ee.electronic_structure_problem.num_particles
        num_spin_orbitals = ee.electronic_structure_problem.num_spin_orbitals
        num_qubits = qubit_op.num_qubits
        log.info(f"Number of particles:  {num_particles}, Number of spin orbitals: {num_spin_orbitals}, Number of qubits: {num_qubits}")
      
        # Setup the ansatz. This ansatz will be used for all future computations.
        if mapper == "QEE":
            init_state = None
            log.info(f"Initial state is state zero for mapper {mapper}.")
        else:
            init_state=ee.set_initial_state(debug=is_debug)
            log.info(f"Initial state set for the mapper {mapper}.")

        ansatz = ee.build_ansatz(num_qubits, init_state, rotations=["ry"], entanglement="cx", entanglement_type="linear", depth=2, debug=is_debug)
        #ansatz = build_custom_ansatz(num_qubits=num_qubits)

        qcex = qce(ansatz, num_qubits)
        if outputpath_exists == True and is_ansatz_printed == False:
            qcex.draw_circuit(in_filename=outputfilepath + '/' + 'ansatz_min_search_' + mapper + '_' + molecule + '.png', is_decompose=True, in_output_type='mpl')
            #ansatz.decompose().draw(output='mpl',filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png')
            log.info(f"Ansatz printed. Check for file ansatz_min_search_{mapper}_{molecule}.png in the output directory.")
            is_ansatz_printed = True

        result = get_vqe_result(qcex=qcex, ansatz=ansatz, qubit_op=qubit_op, is_simulator=is_simulator, debug=is_debug)

        ## interpret VQE result and get optimal parameters
        es_result = ee.electronic_structure_problem.interpret(result)
        if is_debug == True:
            log.debug(es_result)

        total_energy.append(es_result.total_energies.real[0])
    
        log.info(f"Ground state energy for distance {dist} via VQE: {total_energy}")
    
    energies_gs = np.array(total_energy)
    log.info(f"Computing energy values using VQE completed. The values are: {energies_gs}")

    """
    ***************************************************************
    Phase II: In this phase, we use the gradient descent method to identify minimum energy configuration.
    ***************************************************************
    """
    ### Part(A): Compute the gradient
    gradient_gs = np.gradient(energies_gs, distance_list)
    log.info(f"The gradient list values are: {gradient_gs}")

    ### Part(B): Setup cubic spline for interpolation over gradient values
    gradient_cs = CubicSpline(x=distance_list, y=gradient_gs, bc_type='natural')
    log.info(f"The gradient values setup for cubic spline interpolation.")

    ### Part(C): Invoke Gradient Descent
    max_iterations = 100
    tol = 10**(-3) # tolerance limit to stop the computation on finding minimum configuration
    learning_rate = 0.3 # keeping it constant for now

    curr_iter = 1
    log.info(f"Current optimization iteration number: {curr_iter}")
    iter_distance = start_distance + 0.1
    iter_gradient = gradient_cs(iter_distance) # initialize 
    optimal_distance = iter_distance

    while(curr_iter < max_iterations and abs(learning_rate * iter_gradient) >= tol):
        curr_iter = curr_iter + 1
        log.info(f"Current optimization iteration number: {curr_iter}")

        ## identify the next step size
        iter_distance = iter_distance - (learning_rate * iter_gradient)

        ## find interpolated gradient value 
        iter_gradient = gradient_cs(iter_distance)
        log.info(f"Gradient: {iter_gradient}")
        
        ## set potential optimal distance value
        optimal_distance = iter_distance

    # Part (D): Identify interpolated energy at optimal distance value
    energy_cs = CubicSpline(x=distance_list, y=energies_gs, bc_type='natural')
    optimal_energy = energy_cs(x=optimal_distance)

    if(abs(learning_rate * iter_gradient) < tol):
        log.info(f"Found the minimum energy {optimal_energy} at distance {optimal_distance} Angstrom and optimizer iteration count {curr_iter}.")
    elif curr_iter >= max_iterations:
        log.info(f"Processing stopeed as max iteration count of optimizer {curr_iter} reached.")
        log.info(f"The probable minimum energy {optimal_energy} is at distance {optimal_distance} Angstrom.")
    else:
        log.exception(f"Abnormal termination of program!!")

    log.info(f"Program execution completed.")
    log.info(f"****************************************************************")

if __name__ == '__main__':
    main()