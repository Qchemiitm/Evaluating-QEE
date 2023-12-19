"""
The purpose of this module is to estimate the ground state and excited state energies of some key atoms or molecules e.g. H2,LiH,BeH2,C,etc.
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
from EnergyEstimator import EnergyEstimator, variational_eigen_solver, exact_eigen_solver,  plot_energy_graph, single_plot_graph
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

    is_solve_exact = bool(parameters['default']['is_solve_exact'])

    enable_debug = parameters['default']['enable_debug']

    ### Get the individual atom/molecule details 
    molecules = ["H2", "LiH", "C", "BeH2","H2O"]
    try:
        if molecule in molecules:
            multiplicity = parameters[molecule]['multiplicity']
            if molecule == "C":
                distance = None
            else:
                distance = parameters[molecule]['distance']
            # Set the parameters for specific molecules e.g. LiH
            freeze_core = parameters[molecule]['freeze_core'] 
            remove_orbitals = parameters[molecule]['remove_orbitals'] 
            z2symmetry_reduction = parameters[molecule]['z2symmetry_reduction'] 

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
            end_distance = distance[1]
            step_distance = distance[2]

            distance_list = np.arange(start_distance, end_distance, step_distance)
        else:
            raise ValueError(f"Incorrect value set for {distance}. It can be either a list of lists indicating multiple discrete values or a 3-valued list with the first value as start distance, 2nd value as end distance and 3rd value as the increment step of the distance")
    except ValueError as ve:
        log.exception(ve, stack_info=True)
        raise

    log.info(distance_list)
    
    energies_gs = dict() ## ground state energy values 
    energies_gs_classical_comp = dict() ## ground state energy values including exact eigen solvers
    energies_es = dict() ## 1st and 2nd excited state energy values 

    """
    ***************************************************************
    Phase I: In this phase, we execute the Ground state solver and qEOM 
    using QEE as the mapper/transformation.
    ***************************************************************
    """
    
    ## initialize energy estimator object
    log.info(f"Start of processing with QEE transformation ============= ")
    mapper='QEE'
    gs_energy = []
    es_energy = []
    ansatz_printed = False
    apply_z2symmetry = z2symmetry_reduction[mapper]
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
        ee_qee.set_mapper(mapper=mapper, z2symmetry_reduction=apply_z2symmetry, qee_max_allowed_processes = qee_max_allowed_processes, debug=is_debug)
        qubit_op, z2_symmetries = ee_qee.get_hamiltonian_op(debug=is_debug)
        
        ## Verifying z2symmetries
        log.info(f"Z2 Symmetries passed as input: {apply_z2symmetry}")
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
        """
        if  outputpath_exists == True:
        qcex = qce(init_state,ee.electronic_structure_problem.num_spin_orbitals)
        qcex.draw_circuit(outputfilepath + '/' + 'initial_state.png')
        """

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
                                                is_support_aux_operators=False,
                                                quantum_instance=ideal_backend)
        result = execute_vqe.compute_minimum_eigenvalue(qubit_op)
        log.info(f"Hurray! VQE computation for mapper {mapper} is complete.")
        es_result = ee_qee.electronic_structure_problem.interpret(result)
        if is_debug == True:
            log.debug(es_result)
        gs_energy.append(es_result.total_energies.real[0])
        log.info(f"Ground state energy using mapper {mapper} for distance {dist}: {es_result.total_energies.real[0]}")

        #log.info(f"Setting up the ground state solver using mapper {mapper} ... ")
        #gse = GroundStateEigensolver(qubit_converter=ee_qee.qubit_converter, solver=execute_vqe)
        #log.info("Ground state solver set.")
        
        #log.info(f"Solving for ground state solver using mapper {mapper} ... ")
        #result=gse.solve(problem=ee_qee.electronic_structure_problem)
        #es_result = ee_qee.electronic_structure_problem.interpret(result)

        #if is_debug == True:
        #    log.debug(es_result)
        #log.info(f"Solved for ground state solver using mapper {mapper}")

        ###
        ### Solving for the excited state
        ### 
        if es_solver == True:

            log.info(f"Setting up the ground state solver for {mapper} via qEOM... ")
            gse = GroundStateEigensolver(qubit_converter=ee_qee.qubit_converter, solver=execute_vqe)
            log.info(f"Ground state solver set for mapper {mapper}.")

            log.info(f"Solving for excited state via qEOM for {mapper} ... ")
            qeomSolver=QEOM(ground_state_solver=gse, excitations='sd')

            curr_environ_val = os.environ['QISKIT_IN_PARALLEL']
            log.info(f"Current value of QISKIT_IN_PARALLEL environmental variable: {curr_environ_val}")
            os.environ['QISKIT_IN_PARALLEL'] = "True"  # pretends the code already runs in parallel
            result=qeomSolver.solve(ee_qee.electronic_structure_problem)
            os.environ['QISKIT_IN_PARALLEL'] = curr_environ_val

            es_result = ee_qee.electronic_structure_problem.interpret(result)
            if is_debug == True:
                log.debug(es_result)

            es_energy.append((es_result.total_energies.real[1], es_result.total_energies.real[2]))
            log.info(f"1st and 2nd Excited state energies using mapper {mapper} for distance {dist}: {es_result.total_energies.real[1]} and {es_result.total_energies.real[2]}")

    # the energies_gs dictionary is used to compare energies of QEE, JW and P mappers using VQE
    energies_gs[mapper] = gs_energy
    # the energies_gs_classical_comp distionary is used to compare energies of QEE via VQE vs. JW and P mappers using classical eigen solvers
    energies_gs_classical_comp[mapper] = gs_energy
    # the energies_es dictionary is used to hold the excited state energy values. For the comparison, appropriate datasets have to be prepared accordingly.
    energies_es[mapper] = es_energy
    log.info(f"Solved for ground state and excited state energy using mapper {mapper}")

    if len(distance_list) > 1:
        log.info(f"Plotting energy vs. interatomic distance for mapper {mapper} ...")
        energyplot_location = outputfilepath + '/' + 'energyplot_' + mapper + '_' + molecule + '.png'
        plt=single_plot_graph(x_values=distance_list, y_values=gs_energy, plot_title="Energy vs. interatomic distance", x_label="Distance (Angstrom)", y_label="Energy (hartree)", is_annotate=True, debug=is_debug)
        plt.savefig(energyplot_location)
        plt.close()

    ### Identifying the bond distance corresponding to the minimum ground state energy
    min_gsenergy = min(gs_energy)
    min_gsenergy_index = gs_energy.index(min_gsenergy)
    min_gsenergy_dist = distance_list[min_gsenergy_index]
    log.info(f"Bond length for minimum ground state energy for molecule {molecule} using {mapper} is: {min_gsenergy_dist}")
    ### Identifying the excited state energy for the bond length corresponding to the minium ground state energy found in the above step.
    if es_solver == True:
        min_esenergy = es_energy[min_gsenergy_index]
        log.info(f"Excited state energy for bond length of minimum ground state energy for molecule {molecule} using {mapper} is: {min_esenergy}")

    """
    ***************************************************************
    Phase II: In this phase, we execute either the Ground state solver and qEOM 
    using the additional mappers such as Jordan-Wigner, Parity, etc.
    ***************************************************************
    """

    ## initialize energy estimator object
    log.info(f"Starting phase II of computation ...")
    mapper_list = ['JW','P']
    for mapper in mapper_list:
        log.info(f"Start of processing with {mapper} transformation ============= ")
        gs_energy = []
        es_energy = []
        exact_energy = []
        init_state_printed = False
        ansatz_printed = False
        apply_z2symmetry = z2symmetry_reduction[mapper]
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
                # special case of Z2 symmetry for JW and P mappers when dist = 2 Angstrom
                if dist > 1.9 and dist <= 2.0 and len(z2symmetry_reduction[mapper])>0: 
                    if mapper == "JW":
                        apply_z2symmetry = [1,1,1,1,1]
                    elif mapper == "P":
                        apply_z2symmetry = [1,1,1]
                else:
                    apply_z2symmetry = z2symmetry_reduction[mapper]
            else:    
                pass
            log.info(f"Geometry with interatomic distance is: {geometry}")

            ee_map = EnergyEstimator(geometry, multiplicity, freeze_core, remove_orbitals, debug=is_debug)
            ee_map.set_mapper(mapper=mapper, z2symmetry_reduction=apply_z2symmetry, debug=is_debug)
            qubit_op, z2_symmetries = ee_map.get_hamiltonian_op(debug=is_debug)
            
            ## Verifying z2symmetries
            log.info(f"Z2 Symmetries passed as input for mapper {mapper}: {apply_z2symmetry}")
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

            if molecule == "BeH2":
                ansatz = ee_map.build_ucc_ansatz(init_state, depth=5, debug=is_debug)
                """
                if outputpath_exists == True and ansatz_printed == False:
                    qcex = qce(ansatz, num_spin_orbitals)
                    qcex.draw_circuit(in_filename=outputfilepath + '/' + 'ansatz_ucc_' + mapper + '_' + molecule + '.png', is_decompose=True, in_output_type='mpl')
                    #ansatz.decompose().draw(output='mpl',filename=outputfilepath + '/' + 'ansatz_' + mapper + '_' + molecule + '.png')
                    ansatz_printed = True
                    log.info(f"Ansatz printed. Check for file ansatz_ucc_{mapper}_{molecule}.png in the output directory.")
                """
            else:    
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
            result = execute_vqe.compute_minimum_eigenvalue(qubit_op)
            log.info(f"Hurray! VQE computation for mapper {mapper} is complete.")
            es_result = ee_map.electronic_structure_problem.interpret(result)
            if is_debug == True:
                log.debug(es_result)
            gs_energy.append(es_result.total_energies.real[0])
            log.info(f"Ground state energy using mapper {mapper} for distance {dist}: {es_result.total_energies.real[0]}")

            #log.info(f"Setting up the ground state solver for mapper {mapper} ... ")
            #gse = GroundStateEigensolver(qubit_converter=ee_map.qubit_converter, solver=execute_vqe)
            #log.info(f"Ground state solver set.")
            
            #log.info(f"Solving for ground state solver using mapper {mapper} ... ")
            #result=gse.solve(problem=ee_map.electronic_structure_problem)
            #es_result = ee_map.electronic_structure_problem.interpret(result)
            #gs_energy.append(es_result.total_energies.real[0])
            #log.info(f"Ground state energy using mapper {mapper} for distance {distance}: {es_result.total_energies.real[0]}")

            #if is_debug == True:
            #    log.debug(es_result)
            #log.info(f"Solved for ground state solver using mapper {mapper}.")

            # Also invoke the exact eigen solver if parameter set to True
            if is_solve_exact == True:
                exact_eigen_value = exact_eigen_solver(qubit_op)
                exact_energy.append(exact_eigen_value + np.real(es_result.extracted_transformer_energy + es_result.nuclear_repulsion_energy))
                if is_debug == True:
                    print(exact_eigen_value)

            ### Solving for the excited state
            if es_solver == True:
                
                log.info(f"Setting up the ground state solver for mapper {mapper} ... ")
                gse = GroundStateEigensolver(qubit_converter=ee_map.qubit_converter, solver=execute_vqe)
                log.info(f"Ground state solver set for mapper {mapper}.")

                log.info(f"Solving for excited state via qEOM for mapper {mapper} ... ")
                qeomSolver=QEOM(ground_state_solver=gse, excitations='sd')

                os.environ['QISKIT_IN_PARALLEL'] = "True"  # pretends the code already runs in parallel
                result=qeomSolver.solve(ee_map.electronic_structure_problem)
                os.environ['QISKIT_IN_PARALLEL'] = curr_environ_val
                
                es_result = ee_map.electronic_structure_problem.interpret(result)  
                es_energy.append((es_result.total_energies.real[1], es_result.total_energies.real[2]))
                log.info(f"1st and 2nd Excited state energies using mapper {mapper} for distance {dist}: {es_result.total_energies.real[1]} and {es_result.total_energies.real[2]}")

        energies_gs[mapper] = gs_energy
        energies_gs_classical_comp[mapper+'_EX'] = exact_energy
        energies_es[mapper] = es_energy

    if len(distance_list) > 1:
        # plotting ground state energy for QEE, JW and Parity mappers
        log.info(f"Plotting energy vs. interatomic distance for different mappers ...")
        energyplot_comp_location = outputfilepath + '/' + 'energyplot_compare_' + molecule +'.png'
        plt2=plot_energy_graph(distances=distance_list, energies=energies_gs, debug=is_debug)
        plt2.savefig(energyplot_comp_location)
        plt2.close()

        # zooming into plot for ground state energy for QEE, JW and Parity mappers
        # here, we consider interatomic distance 0.3 Angstrom about the minimum energy distance point 
        # identified by QEE
        inset_delta = math.ceil(0.3/step_distance)
        inset_start_index = min_gsenergy_index - inset_delta
        inset_end_index = min_gsenergy_index + inset_delta
        # form the new list for plotting
        inset_distance_list = distance_list[inset_start_index:inset_end_index+1]
        inset_energies_gs = dict()
        for key,val in energies_gs.items():
            log.info(f"Adding data for inset graph for mapper {key} ...")
            inset_energies_gs[key] = val[inset_start_index:inset_end_index+1]
        # let us plot this data
        log.info(f"Zooming into energy vs. interatomic distance plot for different mappers ...")
        energyplot_inset_comp_location = outputfilepath + '/' + 'energyplot_inset_compare_' + molecule +'.png'
        plt3=plot_energy_graph(distances=inset_distance_list, energies=inset_energies_gs, debug=is_debug)
        plt3.savefig(energyplot_inset_comp_location)
        plt3.close()

    # plotting ground state energy for QEE as compared with the exact classical solver
    if len(distance_list) > 1:
        log.info(f"Plotting energy vs. interatomic distance for different mappers with exact eigen solver ...")
        energyplot_comp_location_exact = outputfilepath + '/' + 'energyplot_compare_exact_' + molecule +'.png'
        plt2=plot_energy_graph(distances=distance_list, energies=energies_gs_classical_comp, debug=is_debug)
        plt2.savefig(energyplot_comp_location_exact)
        plt2.close()

    # plotting excited state energy values for QEE, JW and Parity as well as comparing
    # excited state of QEE with ground state energy values of QEE
    if len(distance_list) > 1 and es_solver == True:
        # identify the 1st excited state energy values across different mappers
        energies_es_1st = dict()
        energies_es_1st["QEE"] = [e[0] for e in energies_es["QEE"]]
        for mapper in mapper_list:
            energies_es_1st[mapper] = [e[0] for e in energies_es[mapper]]

        log.info(f"Plotting 1st excited state energy vs. interatomic distance for different mappers ...")
        energyplot_comp_location = outputfilepath + '/' + '1st_excited_energyplot_compare_' + molecule +'.png'
        plt2=plot_energy_graph(distances=distance_list, energies=energies_es_1st, debug=is_debug)
        plt2.savefig(energyplot_comp_location)
        plt2.close()

        # identify the 1st excited state and ground state energy values for QEE
        energies_gs_es_1st = dict()
        energies_gs_es_1st["QEE_GS"] = [g for g in energies_gs["QEE"]]
        energies_gs_es_1st["QEE_ES"] = [e[0] for e in energies_es["QEE"]]

        log.info(f"Plotting energy vs. interatomic distance for different mappers with exact eigen solver ...")
        energyplot_comp_location_exact = outputfilepath + '/' + 'energyplot_compare_gs_es_' + molecule +'.png'
        plt2=plot_energy_graph(distances=distance_list, energies=energies_gs_es_1st, debug=is_debug)
        plt2.savefig(energyplot_comp_location_exact)
        plt2.close()

    if is_debug == True:
        log.debug(f"Ground State values: {energies_gs}")
        log.debug(f"Ground State values with exact classical: {energies_gs_classical_comp}")
        log.debug(f"Excited state values: {energies_es}")

    print("====================================================================")
    if len(distance_list) == 1:
        print(f"The ground state value for molecule {molecule} having bond length {distance_list[0]} using mapper QEE is: {energies_gs['QEE'][0]}")
        for mapper in mapper_list:
            print(f"The ground state value for molecule {molecule} having bond length {distance_list[0]} using mapper {mapper} is: {energies_gs[mapper][0]}")
            print(f"The ground state value for molecule {molecule} having bond length {distance_list[0]} using mapper {mapper} and solved via exact eigen solver is: {energies_gs_classical_comp[mapper+'_EX'][0]}")
    else:
        print(f"The minimum ground state value for molecule {molecule} having bond length {min_gsenergy_dist} using mapper QEE is: {min_gsenergy}")
        for mapper in mapper_list:
            print(f"The ground state value for molecule {molecule} having bond length {min_gsenergy_dist} using mapper {mapper} is: {energies_gs[mapper][min_gsenergy_index]}")
            print(f"The ground state value for molecule {molecule} having bond length {min_gsenergy_dist} using mapper {mapper} and solved via exact eigen solver is: {energies_gs_classical_comp[mapper+'_EX'][min_gsenergy_index]}")

    if es_solver == True:
        print(f"The 1st excited state value for molecule {molecule} having bond length {min_gsenergy_dist} using mapper QEE is: {energies_es['QEE'][min_gsenergy_index][0]}")
        for mapper in mapper_list:
            print(f"The 1st excited state value for molecule {molecule} having bond length {min_gsenergy_dist} using mapper {mapper} is: {energies_es[mapper][min_gsenergy_index][0]}")
        
        print(f"The 2nd excited state value for molecule {molecule} having bond length {min_gsenergy_dist} using mapper QEE is: {energies_es['QEE'][min_gsenergy_index][1]}")
        for mapper in mapper_list:
            print(f"The 2nd excited state value for molecule {molecule} having bond length {min_gsenergy_dist} using mapper {mapper} is: {energies_es[mapper][min_gsenergy_index][1]}")
    
    if len(distance_list) > 1:
        print(f"Identifying the potential depth of the molecule {molecule} ...")
        max_distance_index = len(distance_list)-1
        potential_depth = energies_gs["QEE"][max_distance_index] - min_gsenergy
        print(f"The potential depth of molecule {molecule} when using QEE mapper: {potential_depth}")
        for mapper in mapper_list:
            gs_energy = energies_gs[mapper]
            min_gsenergy_mapper = min(gs_energy)
            min_gsenergy_mapper_index = gs_energy.index(min_gsenergy_mapper)
            potential_depth = energies_gs[mapper][max_distance_index] - energies_gs[mapper][min_gsenergy_mapper_index]
            print(f"The potential depth of molecule {molecule} when using {mapper} mapper: {potential_depth}")

    print("====================================================================")

    log.info(f"Program execution completed.")
    log.info(f"****************************************************************")

if __name__ == '__main__':
    main()