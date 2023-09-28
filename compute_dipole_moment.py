"""
The purpose of this module is to estimate the electric dipole moments of some molecules e.g. H2,LiH,etc.
Author(s): Amit S. Kesari
"""
## importing basic modules
import yaml, os
import numpy as np
import math
## importing qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.algorithms.excited_states_solvers import QEOM

## importing custom modules
from EnergyEstimator import EnergyEstimator, variational_eigen_solver, plot_dipole_moment_graph, single_plot_graph
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

# start of main function
def main():
    log.info("=============================================")
    log.info(f"Start of program ...")
    log.info(f"Checking if output path exists ...")
    outputpath_exists = is_folder_path_exists(outputfilepath)

    log.info(f"Loading parameter file ...")
    ## load the parameter.yaml file
    skip_lines=10
    try:
        with open("parameters.yaml", 'r') as param_stream:
            for i in range(skip_lines):
                _ = param_stream.readline()
            parameters = yaml.safe_load(param_stream)
    except FileNotFoundError as fnf:
        raise
    finally:
        param_stream.close()
    
    log.info(f"paramaters: {parameters}")
    log.info(f"Parameter file read successfully.")

    # Set the geometry of molecule / atom
    molecule = parameters['default']['molecule']

    # Set the parameters for initial state and build the ansatz
    rotations = parameters['default']['rotations']
    entanglement = parameters['default']['entanglement']
    entanglement_type = parameters['default']['entanglement_type']
    depth = parameters['default']['depth']

    # Set the parameters for optimizer
    optimizer_label = parameters['default']['optimizer_label']
    optimizer_maxiter = parameters['default']['optimizer_maxiter']

    # Set the maximum allowed processes for QEE mapper alone
    qee_max_allowed_processes = parameters['default']['qee_max_allowed_processes']
    enable_debug = parameters['default']['enable_debug']

    ### Get the individual atom/molecule details 
    molecules = ["H2", "LiH"]
    try:
        if molecule in molecules:
            multiplicity = parameters[molecule]['multiplicity']
            distance = parameters[molecule]['distance']
            # Set the parameters for specific molecules e.g. LiH
            freeze_core = parameters[molecule]['freeze_core'] 
            remove_orbitals = parameters[molecule]['remove_orbitals'] 
            z2symmetry_reduction = parameters[molecule]['z2symmetry_reduction'] 

            ## ignore the excited state solver for this use case
            es_solver = parameters[molecule]['run_es_solver']
        else:
            raise ValueError(f"Incorrect value for molecule: {molecule}. Value can be only one of {molecules}.")
    except ValueError as ve:
        log.exception(ve, stack_info=True)
        raise

    is_simulator = 1 ## we only run in simulator mode
    is_debug = enable_debug

    log.info(f"Parameters assigned successfully as follows.")
    log.info(f"Molecule: {molecule}")
    log.info(f"Molecule optimization: orbital reduction: {remove_orbitals}, z2symmetry reduction: {z2symmetry_reduction}")
    log.info(f"Interatomic distance: {distance}")
    log.info(f"Ansatz creation parameters: rotations: {rotations}, entanglement: {entanglement}, entanglement type: {entanglement_type}")
    log.info(f"Optimizer label for VQE: {optimizer_label}")
    log.info(f"Running program in debug mode (0 - No, 1 - Yes)?: {enable_debug}")

    # load the qiskit account for execution
    log.info(f"Loading Qiskit account. Please wait as this may take a while ...")
    qce.load_account()
    log.info(f"Qiskit account loaded successfully.")

    try:
        if distance is None or distance == []:
            start_distance = 0
            end_distance = start_distance + 0.01 ### a small increment to enable single iteration
            step_distance = 1000 ### a very large number
                
            distance_list = np.arange(start_distance, end_distance, step_distance)

        elif isinstance(distance[0], list) == True:
            distance_list = [dist for sublist in distance for dist in sublist]
        
        elif len(distance) == 3:
            start_distance = distance[0]
            ## we ensure that the maximum distance for LiH is not beyond 2.5 Angstrom when computing dipole moment
            if molecule == 'LiH':
                end_distance = min(2.5, distance[1])
            else:
                end_distance = distance[1]
            step_distance = distance[2]

            distance_list = np.arange(start_distance, end_distance, step_distance)
        else:
            raise ValueError(f"Incorrect value set for {distance}. It can be either a list of lists indicating multiple discrete values or a 3-valued list with the first value as start distance, 2nd value as end distance and 3rd value as the increment step of the distance")
    except ValueError as ve:
        log.exception(ve, stack_info=True)
        raise

    log.info(distance_list)
    
    dipole_gs = dict() ## dipole moment values

    """
    ***************************************************************
    Phase I: In this phase, we execute the Ground state solver to compute dipole moment
    using QEE as the mapper/transformation.
    ***************************************************************
    """
    
    ## initialize energy estimator object
    log.info(f"Start of processing with QEE transformation ============= ")
    mapper='QEE'
    gs_electronic_dipole_moment = []
    ansatz_printed = False
    for dist in distance_list:
        if molecule == 'H2':
            geometry = [["H", [0, 0, 0]], ["H", [0, 0, dist]]]
        elif molecule == 'LiH':
            geometry = [["Li", [0, 0, 0]], ["H", [0, 0, dist]]]
        elif molecule == 'C':
            geometry = [["C", [0, 0, 0]]]
        elif molecule == 'H2O':
            theta_0 = 104.5 ## angle in degrees
            xdist = math.cos(math.radians(theta_0 / 2)) * dist
            zdist = math.sin(math.radians(theta_0 / 2)) * dist
            geometry = [["O", [0, 0, 0]], ["H", [xdist, 0, zdist]], ["H", [xdist, 0, -zdist]]]
        elif molecule == 'BeH2':
            geometry = [["Be", [0, 0, 0]], ["H", [0, 0, dist]], ["H", [0, 0, -dist]]]
        else:
            pass
        log.info(f"Geometry with interatomic distance is: {geometry}")

        ee_qee = EnergyEstimator(geometry, multiplicity, freeze_core, remove_orbitals, debug=is_debug)
        ee_qee.set_mapper(mapper=mapper, z2symmetry_reduction=z2symmetry_reduction[mapper], qee_max_allowed_processes = qee_max_allowed_processes, debug=is_debug)
        qubit_op, z2_symmetries = ee_qee.get_hamiltonian_op(debug=is_debug)
        
        ## Verifying z2symmetries
        log.info(f"Z2 Symmetries passed as input: {z2symmetry_reduction[mapper]}")
        if z2_symmetries.is_empty():
            log.info(f"No Z2 Symmetries identified for the qubit operator via mapper {mapper}.")
        else:
            log.info(f"Z2 Symmetries identified for the qubit operator via mapper {mapper}.")
            log.info(f"{z2_symmetries}")

        num_particles = ee_qee.electronic_structure_problem.num_particles
        num_spin_orbitals = ee_qee.electronic_structure_problem.num_spin_orbitals
        num_qubits = qubit_op.num_qubits

        #print(qubit_op)
        log.info(f"Number of particles:  {num_particles}, Number of spin orbitals: {num_spin_orbitals}, Number of qubits: {num_qubits}")

        init_state=None
        ansatz = ee_qee.build_ansatz(num_qubits, init_state, rotations, entanglement, entanglement_type, depth, debug=is_debug)
        if outputpath_exists == True and ansatz_printed == False:
            qcex = qce(ansatz, num_spin_orbitals)
            qcex.draw_circuit(in_filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png', is_decompose=True, in_output_type='mpl')
            #ansatz.decompose().draw(output='mpl',filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png')
            ansatz_printed = True
            log.info(f"Ansatz printed. Check for file ansatz_{mapper}_{molecule}.png in the output directory.")

        #execute VQE algorithm for the ideal simulator
        qcex = qce(QuantumCircuit(), num_qubits)
        ## ignore noise model and coupling map as they are not applicable for ideal simulator
        ideal_backend, _, _ = qcex.get_backend(is_simulator=bool(is_simulator), 
                                                                simulator_type='AER_STATEVEVCTOR',
                                                                noise_model_device=None
                                                            )
                                                            
        myoptimizer = EnergyEstimator.get_optimizer(optimizer_label=optimizer_label, 
                                                    maxiter = optimizer_maxiter,
                                                    debug=is_debug)

        log.info(f"Starting phase I of computation ...")
        # execute VQE on ideal/noise-free simulator
        log.info(f"Setting up VQE using mapper {mapper} for ground state solver.")
        execute_vqe = variational_eigen_solver(ansatz, optimizer=myoptimizer, 
                                                is_support_aux_operators=True,
                                                quantum_instance=ideal_backend)
        
        log.info(f"Setting up the ground state solver using mapper {mapper} ... ")
        gse = GroundStateEigensolver(qubit_converter=ee_qee.qubit_converter, solver=execute_vqe)
        log.info("Ground state solver set.")
        
        log.info(f"Solving for ground state solver using mapper {mapper} ... ")
        result=gse.solve(problem=ee_qee.electronic_structure_problem)
        es_result = ee_qee.electronic_structure_problem.interpret(result)
        gs_electronic_dipole_moment.append(es_result.electronic_dipole_moment[0][2])

        if is_debug == True:
            log.debug(es_result)
        log.info(f"Ground state electronic dipole moment using mapper {mapper} for distance {distance}: {es_result.electronic_dipole_moment}")

    dipole_gs[mapper] = gs_electronic_dipole_moment
    log.info(f"Solved for ground state energy using mapper {mapper}")

    if len(distance_list) > 1:
        log.info(f"Plotting dipole moment vs. interatomic distance for mapper {mapper} ...")
        dipoleplot_location = outputfilepath + '/' + 'dipoleplot_' + mapper + '_' + molecule + '.png'
        plt=single_plot_graph(x_values=distance_list, y_values=gs_electronic_dipole_moment, plot_title="Dipole Moment vs. interatomic distance", x_label="Distance (Angstrom)", y_label="Dipole Moment (a.u)", is_annotate=False, debug=is_debug)
        plt.savefig(dipoleplot_location)
        plt.close()
   
    """
    ***************************************************************
    Phase II: In this phase, we compute the dipole moment by executing the Ground state solver
    for the additional mappers such as Jordan-Wigner, Parity, etc.
    ***************************************************************
    """

    ## initialize energy estimator object
    log.info(f"Starting phase II of computation ...")
    mapper_list = ['JW','P']
    for mapper in mapper_list:
        log.info(f"Start of processing with {mapper} transformation ============= ")
        gs_electronic_dipole_moment = []
        init_state_printed = False
        ansatz_printed = False
        for dist in distance_list:
            if molecule == 'H2':
                geometry = [["H", [0, 0, 0]], ["H", [0, 0, dist]]]
            elif molecule == 'LiH':
                geometry = [["Li", [0, 0, 0]], ["H", [0, 0, dist]]]
            elif molecule == 'C':
                geometry = [["C", [0, 0, 0]]]
            elif molecule == 'H2O':
                theta_0 = 104.5 ## angle in degrees
                xdist = math.cos(math.radians(theta_0 / 2)) * dist
                zdist = math.sin(math.radians(theta_0 / 2)) * dist
                geometry = [["O", [0, 0, 0]], ["H", [xdist, 0, zdist]], ["H", [xdist, 0, -zdist]]]
            elif molecule == 'BeH2':
                geometry = [["Be", [0, 0, 0]], ["H", [0, 0, dist]], ["H", [0, 0, -dist]]]
            else:    
                pass
            log.info(f"Geometry with interatomic distance is: {geometry}")

            ee_map = EnergyEstimator(geometry, multiplicity, freeze_core, remove_orbitals, debug=is_debug)
            ee_map.set_mapper(mapper=mapper, z2symmetry_reduction=z2symmetry_reduction[mapper], debug=is_debug)
            qubit_op, z2_symmetries = ee_map.get_hamiltonian_op(debug=is_debug)
            
            ## Verifying z2symmetries
            log.info(f"Z2 Symmetries passed as input for mapper {mapper}: {z2symmetry_reduction[mapper]}")
            if z2_symmetries.is_empty():
                log.info(f"No Z2Symmetries identified for the qubit operator via mapper {mapper}.")
            else:
                log.info(f"Z2Symmetries identified for the qubit operator via mapper {mapper}.")
                log.info(f"{z2_symmetries}")

            num_particles = ee_map.electronic_structure_problem.num_particles
            num_spin_orbitals = ee_map.electronic_structure_problem.num_spin_orbitals
            num_qubits = qubit_op.num_qubits

            #print(qubit_op)
            log.info(f"Number of particles:  {num_particles}, Number of spin orbitals: {num_spin_orbitals}, Number of qubits: {num_qubits}")

            init_state=ee_map.set_initial_state(debug=is_debug)
            if  outputpath_exists == True and init_state_printed == False:
                qcex = qce(init_state, num_spin_orbitals)
                qcex.draw_circuit(outputfilepath + '/' + 'initial_state_' + mapper + '_' + molecule + '.png', in_output_type='mpl')
                init_state_printed = True
                log.info(f"Initial state printed. Check for file initial_state_{mapper}_{molecule}.png in the output directory.")

            ansatz = ee_map.build_ansatz(num_qubits, init_state, rotations, entanglement, entanglement_type, depth, debug=is_debug)
            if outputpath_exists == True and ansatz_printed == False:
                qcex = qce(ansatz, num_spin_orbitals)
                qcex.draw_circuit(in_filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png', is_decompose=True, in_output_type='mpl')
                #ansatz.decompose().draw(output='mpl',filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png')
                ansatz_printed = True
                log.info(f"Ansatz printed. Check for file ansatz_{mapper}_{molecule}.png in the output directory.")

            #execute VQE algorithm for the ideal simulator
            qcex = qce(QuantumCircuit(), num_qubits)
            ## ignore noise model and coupling map as they are not applicable for ideal simulator
            ideal_backend, _, _ = qcex.get_backend(is_simulator=bool(is_simulator), 
                                                                    simulator_type='AER_STATEVEVCTOR',
                                                                    noise_model_device=None
                                                                )
                                                                
            myoptimizer = EnergyEstimator.get_optimizer(optimizer_label=optimizer_label, 
                                                        maxiter = optimizer_maxiter,
                                                        debug=is_debug)

            execute_vqe = variational_eigen_solver(ansatz, optimizer=myoptimizer, 
                                                    quantum_instance=ideal_backend)

            log.info(f"Setting up the ground state solver for mapper {mapper} ... ")
            gse = GroundStateEigensolver(qubit_converter=ee_map.qubit_converter, solver=execute_vqe)
            log.info(f"Ground state solver set.")
            
            log.info(f"Solving for ground state solver using mapper {mapper} ... ")
            result=gse.solve(problem=ee_map.electronic_structure_problem)
            es_result = ee_map.electronic_structure_problem.interpret(result)
            gs_electronic_dipole_moment.append(es_result.electronic_dipole_moment[0][2])

            if is_debug == True:
                log.debug(es_result)
            log.info(f"Ground state electronic dipole moment using mapper {mapper} for distance {distance}: {es_result.electronic_dipole_moment}")

            if is_debug == True:
                log.debug(es_result)
        
        dipole_gs[mapper] = gs_electronic_dipole_moment
        log.info(f"Solved for ground state energy using mapper {mapper}")
        
        
    if len(distance_list) > 1:
        log.info(f"Plotting dipole moment vs. interatomic distance for different mappers ...")
        dipoleplot_comp_location = outputfilepath + '/' + 'dipoleplot_compare_' + molecule +'.png'
        plt2=plot_dipole_moment_graph(distances=distance_list, dipole_values=dipole_gs, debug=is_debug)
        plt2.savefig(dipoleplot_comp_location)
        plt2.close()

    if is_debug == True:
        log.debug(f"Ground State values: {dipole_gs}")

    print("====================================================================")
    if len(distance_list) == 1:
        print(f"The dipole moment for molecule {molecule} having bond length {distance_list[0]} using mapper QEE is: {dipole_gs['QEE'][0]}")
        for mapper in mapper_list:
            print(f"The ground state value for molecule {molecule} having bond length {distance_list[0]} using mapper {mapper} is: {dipole_gs[mapper][0]}")
    
    print("====================================================================")

    log.info(f"Program execution completed.")
    log.info(f"****************************************************************")

if __name__ == '__main__':
    main()