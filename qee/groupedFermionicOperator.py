"""
The purpose of this module is to adapt the original groupedFermioinicOperator code (https://github.com/m24639297/qubit-efficient-mapping) to later versions of qiskit nature, namely 0.4.5.
Author(s): Amit S. Kesari
"""
from qiskit_nature.mappers.second_quantization import FermionicMapper
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli, SparsePauliOp

import numpy as np
from qee import fermionic2QubitMapping as f2QMap
from concurrent.futures import ProcessPoolExecutor
from qiskit.utils import algorithm_globals

# import logging ### this is qiskit logging
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

def kDelta(i, j):
    return 1 * (i == j)

def label2Pauli(s):
    """
    Convert a Pauli string into Pauli object. 
    Note that the qubits are labelled in descending order: 'IXYZ' represents I_3 X_2 Y_1 Z_0
    
    Args: 
        s (str) : string representation of a Pauli term
    
    Returns:
        qiskit.quantum_info.Pauli: Pauli object of s
    """
    
    xs = []
    zs = []
    label2XZ = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    for c in s[::-1]:
        x, z = label2XZ[c]
        xs.append(x)
        zs.append(z)
    return Pauli(z = zs, x = xs)

# Following class updated by Amit S. Kesari for newer version of qiskit (0.45)
class groupedFermionicOperator(FermionicMapper):
    """
    An alternative representation (grouped-operator form) of `qiskit.chemistry.FermionicOperator`.
    Two-electron terms (a_p^ a_q^ a_r a_s) are rearranged into products of (a_i^ a_j).
    (a_n^, a_n are creation and annihilation operators, respectively, acting on the n-th qubit.)
    """
    def __init__(self, num_spin_orbitals, num_electrons, labeling_method=None, mode='rhf', max_allowed_processes=8):
        """
        This class rewrites a `FermionicOperator` into a grouped-operator form stored in `self.grouped_op`.
        The `self.grouped_op` is a dictionary containing information of all one- and two-electron terms.
        For a one-electron term h_pq=val, it is stored as {(p, q): val}.
        For a two-electron term h_pqrs=val, it is first decomposed into products of one-electron terms,
            e.g. a_p^ a_q^ a_r a_s = kDelta(q, r) * (a_p^ a_s) - (a_p^ a_r) * (a_q^ a_s).
        In case (p, r) is not an allowed transition(due to spin restrictions etc), a_r and a_s 
        can be exchanged with an extra minus sign (handled by `parity` in the code.)
        Finally, all two-electron terms can be decomposed into products of ALLOWED one-electron terms.
        
        The `mapping` is used to convert fermionic operators into qubit operators(typically Pauli terms).
        It is a dictionary whose keys are indices of allowed transitions (e.g. (p, q) if a_p^ a_q is allowed) 
        and values are the Pauli term corresponding to a_p^ a_q.
        
        Args:
            ferOp (qiskit.chemistry.FermionicOperator): second-quantized fermionic operator
            num_electron (int): number of electron in the system
            labeling_method (function):
                It maps each electron occupation configuration to a qubit computational basis state.
                (Please refer to docString of fermionic2QuantumMapping for details)
            mode (str): it should be either 'rhf' or 'uhf'.
                'rhf' mode:
                    Same number of spin-up and spin-down electrons. 
                    Their configurations are encoded to qubits independently.
                    (# of qubits = 2 * (ceil(log2(num_of_config(num_electron/2)))))
                'stacked_rhf' mode:
                    Same number of spin-up and spin-down electrons.
                    All allowed configurations are encoded together
                    (# of qubits = ceil(2 * log2(num_of_config(num_electron/2))))
                'uhf' mode:
                    No restriction on spin conservation. All configuration are encoded to qubits.
                    (# of qubits = ceil(log2(num_of_config(num_electron))))
        """
        super().__init__()
        self.grouped_op = {}
        self.THRESHOLD = 1e-6
        self.num_qubits = None ## needs to be set during map operation 
        self.mapping = {}
        self.num_spin_orbitals = num_spin_orbitals
        self.num_electrons = num_electrons
        self.labeling_method=labeling_method, 
        self.mode = mode
        #self.qubitOp = None ## needs to be set during map operation
        self.cpu_count = min(algorithm_globals.num_processes, max_allowed_processes)
        #self.fermionic_hamiltonian = fermionic_hamiltonian
        
        # h1, h2 = np.copy(ferOp.h1), np.copy(ferOp.h2)
        # it1 = np.nditer(h1, flags=['multi_index'])
        # it2 = np.nditer(h2, flags=['multi_index'])
        # for h in it1:
        #     key = it1.multi_index
        #     self._add_an_h1(h, key)
        # for h in it2:
        #     key = it2.multi_index
        #     self._add_an_h2(h, key)

    def set_groupedFermionicOp(self, fermionic_hamiltonian):
        """
        This sets up the groupedOperator
        """
        ferOp_h1, ferOp_h2 = self.get_h1_h2(fermionic_hamiltonian=fermionic_hamiltonian, num_spin_orbitals=self.num_spin_orbitals)
        self.mapping, self.num_qubits = \
                        f2QMap.fermionic2QubitMapping(num_so = ferOp_h1.shape[0], #ferOp.modes
                                               num_e = self.num_electrons,
                                               mode = self.mode
                                               #labeling_method = self.labeling_method
                                              )
        self.set_ferOp(ferOp_h1, ferOp_h2)

    def set_ferOp(self, ferOp_h1, ferOp_h2):
        h1, h2 = np.copy(ferOp_h1), np.copy(ferOp_h2)
        it1 = np.nditer(h1, flags=['multi_index'])
        it2 = np.nditer(h2, flags=['multi_index'])
        for h in it1:
            key = it1.multi_index
            self._add_an_h1(h, key)
        for h in it2:
            key = it2.multi_index
            self._add_an_h2(h, key)
        
    def _add_an_h1(self, coef, pq):
        """
            Add a single one-electron term into the grouped operator.  
            
            Args:
                coef (complex) : value of one-electron integral
                pq (tuple(int, int)): index of the one-electron term
        """
        if(abs(coef) < self.THRESHOLD): return 
        if pq in self.grouped_op.keys():
            self.grouped_op[pq] = self.grouped_op[pq] + coef
        else:
            # multiplied by 1 to convert array data type to complex
            self.grouped_op[pq] = coef * 1
    
    def _add_an_h2(self, coef, pqrs):
        """
            Add a single two-electron term into the grouped operator. 
            
            Args:
                coef (complex) : value of two-electron integral
                pqrs (tuple(int, int, int, int)): index(in chemist notation) of the two-electron term
        """
        if(abs(coef) < self.THRESHOLD): return 
        parity = 1
        
        ## Note that in FermionicOperator, index (p,q,r,s) represents a_p^ a_r^ a_s a_q: chemist notation
        ## Here I use (p,q,r,s) to represent a_p^ a_q^ a_r a_s: physicist notation
        ## Thus the re-ordering of h-indices is needed
        
        ## In our code, we use the physicist notation and hence below line has been
        ## replaced by subsequent line - Amit S. Kesari
        #p, s, q, r = pqrs
        p, q, r, s = pqrs

        ## Handle the exchange of a_r a_s if direct transformation will give illegal transitions
        if (((p, r) not in self.mapping.keys()) and ((r,p) not in self.mapping.keys())):
            r, s = s, r
            parity = -1
            log.debug(f"Change to: {(p,q,r,s)}")
            
        ## a_p^ a_q^ a_r a_s = kDelta(q, r) * (a_p^ a_s) - (a_p^ a_r) * (a_q^ a_s)
        self._add_an_h1(parity * coef * kDelta(q, r), (p, s))
        mut_key = ((p, r), (q, s))
        if mut_key in self.grouped_op.keys():
            self.grouped_op[mut_key] -= coef * parity
        else:
            self.grouped_op[mut_key] = -coef * parity
    
    def get_qubit_Op(self, group_op_items):
        """
        Get qubit operator from the transition and weights from group operators. A helper function of to_paulis
            Args:
                    group_op_items (group-operator form) : a partial list of key-value pairs from self.group_op
            Returns:
                    (qiskit.aqua.operators.WeightedPauliOperator) : qubit operator transformed based on `self.mapping`
        """
        log.info("One process started ...")
        mapping = self.mapping
        num_qubits = self.num_qubits
        # Following original code commented by Amit S. Kesari to support qiskit 0.45 version
        """
        qubitOp = []
        for k, w in group_op_items:
            if np.ndim(k) == 1:  ## one-e-term
                qubitOp += (w * mapping[k])
            elif np.ndim(k) == 2:  ## 2-e-term
                k1, k2 = k
                qubitOp += (w * mapping[k1] * mapping[k2])
            else:
                raise ValueError('something wrong')
        """
        # Following code added by Amit S. Kesari to support qiskit 0.45 version
        qubitOp = PauliSumOp(SparsePauliOp(Pauli("I" * num_qubits), coeffs=0))
        for k, w in group_op_items:
            tempSumOp = PauliSumOp(SparsePauliOp(Pauli("I" * num_qubits), coeffs=0))
            try:
                if np.ndim(k) == 1: ## one-e-term
                    for item in mapping[k]:
                        tempSumOp = tempSumOp.add(item.mul(w))
                elif np.ndim(k) == 2: ## 2-e-term
                    k1, k2 = k
                    tempPauliOpList = []
                    for item_k1 in mapping[k1]:
                        tempPauliOpList = [item_k1.compose(x) for x in mapping[k2]]
                        for p in tempPauliOpList:
                            tempSumOp = tempSumOp.add(p.mul(w))
                    tempSumOp = tempSumOp.reduce(rtol=self.THRESHOLD)
                else:
                    raise ValueError(f"ERROR: Only 1-body and 2-body electronic integrals can be present. Instead, found dimension {np.ndim(k)}")
            except ValueError as ve:
                log.exception(ve)
                raise

            # Finally, add all the accumulated operators
            qubitOp = qubitOp.add(tempSumOp)
        log.info("One process completed.")
        return qubitOp

    def to_paulis(self):
        """
        Convert the grouped fermionic operator into qubit operators (sum of Pauli terms)
            Args:
                    cpu_count (int) : number of CPUs (processes) to handle this function
            Returns:
                    (qiskit.aqua.operators.WeightedPauliOperator) : qubit operator transformed based on `self.mapping`
        """
        try: 
            log.info(f"Number of processes: {self.cpu_count}")
            if self.cpu_count == 1:
                qubitOp = self.get_qubit_Op(self.grouped_op.items())
                qubitOp = qubitOp.reduce(rtol=self.THRESHOLD)
            elif self.cpu_count > 1:
                arguments = []
                group_op_items_per_cpu = len(self.grouped_op.items()) // self.cpu_count + 1
                prev = 0
                for i in range(self.cpu_count):
                    if i == self.cpu_count - 1:
                        arguments.append(list(self.grouped_op.items())[prev:])
                        break
                    end = group_op_items_per_cpu * (i + 1)
                    arguments.append(list(self.grouped_op.items())[prev:end])
                    prev = end
                with ProcessPoolExecutor() as executor:
                    results = executor.map(self.get_qubit_Op, arguments)
                    ## below line commented by Amit S. Kesari to support qiskit version 0.45
                    ## and instead replaced with PauliSumOp object
                    #qubitOp = PauliOp(paulis=[])
                    qubitOp = PauliSumOp(SparsePauliOp(Pauli("I" * self.num_qubits), coeffs=0))
                    for result in results:
                        qubitOp += result
                ## below line commented by Amit S. Kesari to support qiskit version 0.45        
                #qubitOp.chop(threshold=self.THRESHOLD)
                qubitOp = qubitOp.reduce(rtol=self.THRESHOLD)
            else:
                raise ValueError('CPU count has to be greater or equal to 1')
        except ValueError as ve:
            log.exception(ve)
            raise
        return qubitOp

    @classmethod
    def mysort(cls, str):
        lst = str.split(' ')
        plus = []
        minus = []
        for i in lst:
            if i.startswith('+_'):
                plus.append(i)
            elif i.startswith('-_'):
                minus.append(i)
        newlst = plus + minus
        new_str = ' '.join(newlst)
        return(new_str)

    ### Method included by Amit S. Kesari
    def get_h1_h2(self, fermionic_hamiltonian, num_spin_orbitals):
        """
        This method takes in a fermionic operator and splits into h1 (one body integral)
        and h2 (two body integral)
        """
        
        electron_integrals = fermionic_hamiltonian.to_list(display_format='sparse')
        #log.debug(f"Sparse electron integrals: {electron_integrals}")
        dim = num_spin_orbitals
        h1 = np.zeros(shape=(dim, dim),dtype=complex)
        h2 = np.zeros(shape=(dim, dim, dim, dim),dtype=complex)
        try:
            for integral in electron_integrals:
                op_orig = integral[0]
                coeff = integral[1]
                op = groupedFermionicOperator.mysort(op_orig)
                op_string = op.replace('+_','').replace('-_','')
                # get the tuple of integers
                try:
                    op_tuple = tuple(map(int, op_string.split(' ')))
                except Exception as ex:
                    log.warning(f"Ignored exception for operator string {op_string}: {ex}")
                    continue

                # if length = 2 => 1e integral; if length = 4 => 2e integral
                if len(op_tuple) == 2:
                    h1[op_tuple]=coeff
                elif len(op_tuple) == 4:
                    h2[op_tuple]=coeff
                else:
                    raise ValueError(f"Incorrect length for 1e or 2e integral: {len(op_tuple)}. Dimension can be either 2 or 4 only.")
        except ValueError as ex:
            log.exception(ex)
            raise
        #log.debug(f"1-body electron integrals: {h1}")
        #log.debug(f"2-body electron integrals: {h2}")
        return(h1, h2)

    def map(self, fermionic_hamiltonian):
        """
        Implements map method of FermionicMapper class
        """
        log.info(f"QEE map method invoked ... ")
        #log.debug(f"Input Electron integrals: {fermionic_hamiltonian}")
        self.grouped_op = {}
        self.mapping = {}
        self.set_groupedFermionicOp(fermionic_hamiltonian)
        qubitOp = self.to_paulis()
        log.info(f"QEE map method completed.")
        return(qubitOp)
