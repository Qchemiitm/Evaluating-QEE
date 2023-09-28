from qiskit.algorithms import VQE
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

# Following class defined by Amit S. Kesari so as to disable support for auxiliary operators
class VQENoAuxOperators(VQE):

    def __init__(
        self,
        ansatz = None, #: QuantumCircuit | None = None,
        optimizer = None, #: Optimizer | Minimizer | None = None,
        initial_point = None, #: np.ndarray | None = None,
        gradient = None, #: GradientBase | Callable | None = None,
        expectation = None, #: ExpectationBase | None = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback = None, #: Callable[[int, np.ndarray, float, float], None] | None = None,
        quantum_instance = None #: QuantumInstance | Backend | None = None
        ):
        log.info(f"Initialising VQENoAuxOperators ... ")
        super().__init__(ansatz=ansatz, 
                         optimizer=optimizer,
                         initial_point=initial_point,
                         gradient=gradient,
                         expectation=expectation,
                         include_custom=include_custom,
                         max_evals_grouped=max_evals_grouped,
                         callback=callback,
                         quantum_instance=quantum_instance
                         )

    @classmethod
    def supports_aux_operators(cls) -> bool:
        log.info(f"Custom VQE called with support for auxiliary operators turned off.")
        return False