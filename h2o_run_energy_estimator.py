"""
The purpose of this module is to estimate energy for H2O molecule by changing both the bond length and the angle
H2O is a non-linear molecule and hence the bond angle also plays an important role.
Author(s): Amit S. Kesari
"""
## importing basic modules
import os
import numpy as np
import math
## importing qiskit modules
from qiskit import Aer, IBMQ 
# from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
# from qiskit.opflow import StateFn, CircuitStateFn
# from qiskit.compiler import transpile

## importing custom modules
from EnergyEstimator import EnergyEstimator, variational_eigen_solver, exact_eigen_solver
from QcExecution import QcExecution as qce
from logconfig import get_logger

## define some global variables
qee_max_allowed_processes = 16
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

# setup H2O molecule 
def setup_h2o_molecule(distance, theta, debug=False) -> EnergyEstimator:
    ## first setup geometry
    xdist = math.cos(math.radians(theta / 2)) * distance
    zdist = math.sin(math.radians(theta / 2)) * distance
    geometry = [["O", [0, 0, 0]], ["H", [xdist, 0, zdist]], ["H", [xdist, 0, -zdist]]]
    log.info(f"The geometry of molecule H2O is: {geometry}")

    ## setup additional parameters such as freeze_core and remove_orbitals
    multiplicity = 1
    freeze_core = 1
    remove_orbitals = None
    ## return energy estimator object
    ee = EnergyEstimator(geometry, multiplicity, freeze_core, remove_orbitals, debug=debug)
    log.info(f"H2O molecule setup for distance {distance} and bond angle {theta}.")
    return (ee)

# identify qubit hamiltonian 
def h2o_qubit_hamiltonian(ee, apply_z2symmetry, mapper="QEE", debug=False):
    # set QEE mapper
    if mapper == "QEE":
        ee.set_mapper(mapper=mapper, z2symmetry_reduction=None, qee_max_allowed_processes = qee_max_allowed_processes, debug=debug)

        # get hamiltonian operator
        qubit_op, _ = ee.get_hamiltonian_op(debug=debug)

    else:
        ee.set_mapper(mapper=mapper, z2symmetry_reduction=apply_z2symmetry, debug=debug)
    
        # get hamiltonian operator
        qubit_op, z2_symmetries = ee.get_hamiltonian_op(debug=debug)
        ## Verifying z2symmetries
        log.info(f"Z2 Symmetries passed as input for mapper {mapper}: {apply_z2symmetry}")
        if z2_symmetries.is_empty():
            log.info(f"No Z2Symmetries identified for the qubit operator via mapper {mapper}.")
        else:
            log.info(f"Z2Symmetries identified for the qubit operator via mapper {mapper}.")
            log.info(f"{z2_symmetries}")

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

    energies_gs = dict() ## ground state energy values 

    """
    ***************************************************************
    Phase I: In this phase, we compute energy values for different inter-atomic distances
    using VQE and mapper QEE.
    ***************************************************************
    """

    # Identify the structure of H2O molecule and get qubit hamiltonian
    molecule = "H2O"
    start_distance = 0.7
    end_distance = 1.2
    step_size = 0.1
    is_ansatz_printed = False

    distance_list = np.arange(start=start_distance,stop=end_distance,step=step_size)
    total_energy = []
    total_energy_theta = []

    for dist in distance_list:

        start_theta = 102
        end_theta = 107
        theta_size = 0.5

        theta_list = np.arange(start=start_theta,stop=end_theta,step=theta_size)
        gs_energy = []

        for theta in theta_list:
            ee = setup_h2o_molecule(distance=dist, theta=theta, debug=is_debug)
            qubit_op = h2o_qubit_hamiltonian(ee, apply_z2symmetry=None, mapper=mapper, debug=is_debug)

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

            qcex = qce(ansatz, num_qubits)
            if outputpath_exists == True and is_ansatz_printed == False:
                qcex.draw_circuit(in_filename=outputfilepath + '/' + 'ansatz_energy_' + mapper + '_' + molecule + '.png', is_decompose=True, in_output_type='mpl')
                #ansatz.decompose().draw(output='mpl',filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png')
                log.info(f"Ansatz printed. Check for file ansatz_energy_{mapper}_{molecule}.png in the output directory.")
                is_ansatz_printed = True

            result = get_vqe_result(qcex=qcex, ansatz=ansatz, qubit_op=qubit_op, is_simulator=is_simulator, debug=is_debug)

            ## interpret VQE result and get optimal parameters
            es_result = ee.electronic_structure_problem.interpret(result)
            if is_debug == True:
                log.debug(es_result)

            gs_energy.append(es_result.total_energies.real[0])

            log.info(f"Ground state energy for distance {dist} and angle {theta} via VQE: {es_result.total_energies.real[0]}")

        ### Identifying the theta corresponding to the minimum ground state energy for a given distance
        min_gsenergy = min(gs_energy)
        min_gsenergy_index = gs_energy.index(min_gsenergy)
        min_gsenergy_theta = theta_list[min_gsenergy_index]
        log.info(f"Minimum energy and theta for molecule {molecule} and distance {dist} using {mapper} is: {(min_gsenergy, min_gsenergy_theta)}")
        
        total_energy.append(min_gsenergy)
        total_energy_theta.append(min_gsenergy_theta)

    log.info(f"Computing energy values using VQE completed. The energy values are: {np.array(total_energy)}")
    log.info(f"The corresponding theta values are: {np.array(total_energy_theta)}")

    ### Identifying the bond distance corresponding to the minimum ground state energy
    min_gsenergy = min(total_energy)
    min_gsenergy_index = total_energy.index(min_gsenergy)
    min_gsenergy_theta = total_energy_theta[min_gsenergy_index]
    min_gsenergy_dist = distance_list[min_gsenergy_index]
    energies_gs[mapper] = min_gsenergy

    print("====================================================================")
    log.info(f"Minimum energy for molecule {molecule} for {mapper} is: {min_gsenergy}")
    log.info(f"Minimum energy bond length for molecule {molecule} for {mapper} is: {min_gsenergy_dist}")
    log.info(f"Minimum energy bond angle for molecule {molecule} for {mapper} is: {min_gsenergy_theta}")
    print("====================================================================")

    """
    ***************************************************************
    Phase II: In this phase, we execute the Ground state solver 
    using the additional mappers such as Jordan-Wigner, Parity, etc.
    ***************************************************************
    """

    ## initialize energy estimator object
    log.info(f"Starting phase II of computation ...")
    mapper_list = ['JW','P']
    for mapper in mapper_list:
        log.info(f"Start of processing with {mapper} transformation ============= ")
        init_state_printed = False
        is_ansatz_printed = False
        if mapper == 'JW':
            apply_z2symmetry = [1,1,1,1]
        elif mapper == 'P':
            apply_z2symmetry = [1,1]

        ee = setup_h2o_molecule(distance=min_gsenergy_dist, theta=min_gsenergy_theta, debug=is_debug)
        qubit_op = h2o_qubit_hamiltonian(ee, apply_z2symmetry=apply_z2symmetry, mapper=mapper, debug=is_debug)

        # setting the appropriate parameter values
        num_particles = ee.electronic_structure_problem.num_particles
        num_spin_orbitals = ee.electronic_structure_problem.num_spin_orbitals
        num_qubits = qubit_op.num_qubits
        log.info(f"Number of particles:  {num_particles}, Number of spin orbitals: {num_spin_orbitals}, Number of qubits: {num_qubits}")
        
        # Setup the ansatz. This ansatz will be used for all future computations.
        init_state=ee.set_initial_state(debug=is_debug)
        log.info(f"Initial state set for the mapper {mapper}.")
        if  outputpath_exists == True and init_state_printed == False:
            qcex_init = qce(init_state, num_spin_orbitals)
            qcex_init.draw_circuit(outputfilepath + '/' + 'initial_state_' + mapper + '_' + molecule + '.png', in_output_type='mpl')
            init_state_printed = True
            log.info(f"Initial state printed. Check for file initial_state_{mapper}_{molecule}.png in the output directory.")

        ansatz = ee.build_ansatz(num_qubits, init_state, rotations=["ry"], entanglement="cx", entanglement_type="linear", depth=2, debug=is_debug)

        qcex = qce(ansatz, num_qubits)
        if outputpath_exists == True and is_ansatz_printed == False:
            qcex.draw_circuit(in_filename=outputfilepath + '/' + 'ansatz_energy_' + mapper + '_' + molecule + '.png', is_decompose=True, in_output_type='mpl')
            #ansatz.decompose().draw(output='mpl',filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png')
            log.info(f"Ansatz printed. Check for file ansatz_energy_{mapper}_{molecule}.png in the output directory.")
            is_ansatz_printed = True

        result = get_vqe_result(qcex=qcex, ansatz=ansatz, qubit_op=qubit_op, is_simulator=is_simulator, debug=is_debug)

        ## interpret VQE result and get optimal parameters
        es_result = ee.electronic_structure_problem.interpret(result)
        if is_debug == True:
            log.debug(es_result)
        
        energies_gs[mapper] = es_result.total_energies.real[0]
        log.info(f"Ground state energy using mapper {mapper} for distance {min_gsenergy_dist} and angle {min_gsenergy_theta}: {es_result.total_energies.real[0]}")

        # Also invoke the exact eigen solver if parameter set to True
        exact_eigen_value = exact_eigen_solver(qubit_op)
        energies_gs[mapper+'_EX'] = exact_eigen_value + np.real(es_result.extracted_transformer_energy + es_result.nuclear_repulsion_energy)
        if is_debug == True:
            print(exact_eigen_value)

    print("====================================================================")
    print(f"The minimum ground state value for molecule {molecule} having bond length {min_gsenergy_dist} and bond angle {min_gsenergy_theta} using mapper QEE is: {min_gsenergy}")
    for mapper in mapper_list:
        print(f"The ground state value for molecule {molecule} having bond length {min_gsenergy_dist} and bond angle {min_gsenergy_theta} using mapper {mapper} is: {energies_gs[mapper]}")
        print(f"The ground state value for molecule {molecule} having bond length {min_gsenergy_dist} and bond angle {min_gsenergy_theta} using mapper {mapper} and solved via exact eigen solver is: {energies_gs[mapper+'_EX']}")
    print("====================================================================")

if __name__ == '__main__':
    main()    