# jarvis/verification/input_vector_generator.py
# InputVectorGenerator -- produces the fixed input matrix for the harness.
# Authority: DVH Implementation Blueprint v1.0.0 Sections 5 and 7.
#
# NIC-01: No arithmetic operates on production data paths.
# NIC-05: No control flow of the production Risk Engine is altered.
# EEP-07: No random number generation. Matrix is fully deterministic.

from typing import List
from jarvis.verification.data_models.input_vector import InputVector
from jarvis.verification.vectors.vector_definitions import INPUT_MATRIX, VECTOR_COUNT


class InputVectorGenerator:
    """
    Returns the fixed, ordered input matrix from vector_definitions.py.

    The matrix is version-controlled. No vector is generated at runtime.
    No sampling is performed. The matrix is identical on every harness run
    for a given harness version (Section 7).
    """

    def generate(self) -> List[InputVector]:
        """
        Return the complete ordered list of InputVector instances.

        Execution order follows Section 7.2:
          G-VOL, G-DD, G-MU, G-RP, G-JM, G-CR, G-BC, G-EX
        Within each group: ascending numeric suffix order.
        """
        return list(INPUT_MATRIX)

    def vector_count(self) -> int:
        """Return total number of vectors in the matrix."""
        return VECTOR_COUNT
