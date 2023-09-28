#!/usr/bin/env python
# coding: utf-8

"""
This module contains the basic routines to run quantum programs on simulator, IBM Q machine or 
using state tomography.
Author(s): Amit S. Kesari
"""
from qiskit import QuantumCircuit, QuantumRegister, transpile, Aer, IBMQ
from qiskit.visualization import plot_histogram, matplotlib, circuit_drawer, latex, plot_state_city
from qiskit.tools.monitor import job_monitor
from qiskit_experiments.library import StateTomography
from qiskit.providers.ibmq import least_busy
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo, FakeAthens, FakeAlmaden
from qiskit_aer.backends import AerSimulator, QasmSimulator
import os
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

# define dictionary for backend devices
"""
IMP: 
1. QasmSimulator.from_backend does not currently support V2 nosiy sumulator devices and hence have not been included in the devices dictionary.
2. Some noisy simulator devices e.g. FakeManila throw 'tkinter' error during transpilation and hence are not included as well. Most probably, this is becauase transpiled errors are generated in a multi-threaded manner. 
"""
devices = {'FAKEVIGO': FakeVigo(),
           'FAKEATHENS': FakeAthens(),
           'FAKEALMADEN': FakeAlmaden()
          }

# define class qc_execution
class QcExecution:
    # define __init__ function / constructor
    def __init__(self, quantum_circuit, qubit_num):
        self.qc = quantum_circuit
        self.q_num = qubit_num

    @classmethod
    def load_account(cls):
        IBMQ.load_account()

    def get_noise_model(self, noise_model_device=None):
        try:

            if noise_model_device != None:
                device_backend = devices.get(noise_model_device.upper())
                if device_backend is None:
                    raise Exception(f"Noise model device label can only be one of {list(devices.keys())}")
                
                device = QasmSimulator.from_backend(device_backend)
                coupling_map = device.configuration().coupling_map
                noise_model = NoiseModel.from_backend(device)
            else:
                raise Exception("No noise model defined.")

        except Exception as e:
            log.exception(e, stack_info=True)
            raise
        else:
            return(device_backend, noise_model, coupling_map)    
            
    def get_backend(self, is_simulator=True, simulator_type='AER', noise_model_device=None):        
        try:
            noise_model = None
            coupling_map = None
            
            if is_simulator == False:
                # Get quantum computer backend
                provider = IBMQ.get_provider(hub = 'ibm-q')
                #provider.backends()
                #backend = provider.get_backend('ibm_nairobi')
                backend = least_busy(provider.backends
                                    (filters = lambda x:x.configuration().n_qubits >= self.q_num                                      and not x.configuration().simulator
                                        and x.status().operational==True)
                            )
                if backend is None:
                    raise Exception("No IBM Q backend avaiable for the required criteria.")

            elif is_simulator == True:
                if simulator_type == 'AER_STATEVEVCTOR':
                    backend = Aer.get_backend('aer_simulator_statevector')
                elif simulator_type == 'AER':
                    if noise_model_device is None:
                        backend = Aer.get_backend('aer_simulator')
                    else:
                        backend, noise_model, coupling_map = self.get_noise_model(noise_model_device)
                elif simulator_type == 'QASM':
                    backend = Aer.get_backend('qasm_simulator') 
        except Exception as e:
            log.exception(e, stack_info=True)
            print(e)
            raise
        else:
            return(backend, noise_model, coupling_map)

    # Run the quantum circuit on a statevector simulator backend or actual quantum computer
    def execute_program(self, backend, is_simulator, is_state_tomography):
        max_q_num = self.q_num if self.q_num > 5 else 5
        try:
            if is_simulator not in [0,1]:
                raise ValueError("Value for simulator execution can only be 0 or 1")
        except ValueError as ve:
            log.exception(ve, stack_info=True)
            print(ve)
            raise
        else:
            if is_simulator == 1:
                try:
                    if is_state_tomography not in [0,1]:
                        raise ValueError("Value for state tomography execution can only be 0 or 1")
                except ValueError as ve:
                    log.exception(ve, stack_info=True)
                    print(ve)
                    raise
                else:
                    if is_state_tomography == 0:
                        # Transpile
                        self.qc = transpile(self.qc, backend)

                        # Run the quantum program and get result
                        job = backend.run(self.qc)
                        result = job.result()
                    elif is_state_tomography == 1:
                        # Transpile
                        self.qc = transpile(self.qc, backend)

                        # Setup the QST experiment and execute
                        qstexp = StateTomography(qc)
                        job = qstexp.run(backend, seed_simulation=100).block_for_results()
                        result = job.analysis_results("state")
            
            elif is_simulator == 0:
                # Get quantum computer backend and transpile                
                qc = transpile(qc, backend)
                # Run the quantum program and get result
                job = backend.run(qc,shots = 1024)
                job_monitor(job)
                result = job.result()

        finally:
            return result
    
    # draw the circuit with output to a file - overloaded method
    def draw_circuit(self, in_filename=None, is_decompose=False, in_output_type='latex'):
        #in_filename = '/home/amit/Downloads/asktest.png'
        if in_filename is None:
            if is_decompose:
                self.qc.decompose().draw(output=in_output_type)
            else:
                self.qc.draw(output=in_output_type)
        else:
            v_file_dir = in_filename.rpartition("/")[0]
            try:
                if not os.path.exists(v_file_dir):
                    raise NameError("Filepath does not exist: " + v_file_dir)
            except NameError as ne:
                log.exception(ne, stack_info=True)    
                print(ne)
                raise
            else:
                if is_decompose:
                    self.qc.decompose().draw(output=in_output_type, filename=in_filename)
                else:
                    self.qc.draw(output=in_output_type, filename=in_filename)
        
        #return out

    # Print the output
    def print_result(self, plot_title, result, in_filename=None):
        try:
            qcoutputstate = result.get_statevector(self.qc)
            print(qcoutputstate)
            print(result.get_counts(self.qc))
            count = result.get_counts(self.qc)
        except Exception as ex:
            log.exception(ex, stack_info=True)
            raise(ex)
        else:
            if in_filename is None:
                out = plot_histogram(count, title=plot_title, filename=in_filename)
            else:
                v_file_dir = in_filename.rpartition("/")[0]
                try:
                    if not os.path.exists(v_file_dir):
                        raise NameError("Filepath does not exist: " + v_file_dir)
                except NameError as ne:
                    log.exception(ne, stack_info=True)
                    print(ne)
                    raise
                else:
                    out = plot_histogram(count, title=plot_title, filename=in_filename)
        finally:
            return(out)
  