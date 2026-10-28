import tempfile
from pathlib import Path
import unittest

from qiskit.circuit import IfElseOp

from Model1.Experiment3 import draw_dynamic_lie_trotter_steps as diagram


class DynamicStepDiagramTests(unittest.TestCase):
    def test_three_steps_reuse_two_ancillas_and_one_classical_bit(self):
        circuit = diagram.build_display_circuit(7)
        self.assertEqual(circuit.num_qubits, 9)
        self.assertEqual(circuit.num_clbits, 1)
        self.assertEqual([register.name for register in circuit.qregs], ["system", "ancilla_L", "ancilla_H"])
        self.assertEqual(circuit.count_ops()["barrier"], 3)
        self.assertEqual(circuit.count_ops()["measure"], 3)
        self.assertEqual(circuit.count_ops()["reset"], 6)
        self.assertEqual(circuit.count_ops()["if_else"], 3)
        for instruction in circuit.data:
            if isinstance(instruction.operation, IfElseOp):
                self.assertEqual(instruction.operation.condition, (circuit.clbits[0], 0))

    def test_site_labels_match_bond_labels_and_boundary_interactions(self):
        circuit = diagram.build_display_circuit(7)
        system, low, high = circuit.qregs
        bonds = []
        for instruction in circuit.data:
            label = instruction.operation.label or ""
            if label.startswith("H_"):
                indices = tuple(map(int, label.split("[")[1][:-1].split(",")))
                self.assertEqual(instruction.qubits, tuple(system[i] for i in indices))
                self.assertEqual("even" in label, indices[0] % 2 == 0)
                bonds.append(indices)
            elif label == "K_1":
                self.assertEqual(instruction.qubits, (system[0], low[0]))
            elif isinstance(instruction.operation, IfElseOp):
                self.assertEqual(instruction.qubits, (system[-1], high[0]))
            elif instruction.operation.name == "measure":
                self.assertEqual(instruction.qubits, (low[0],))
        self.assertEqual(len(bonds), 3 * 6)

    def test_draw_generates_png_and_vector_pdf_without_running_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = diagram.draw_steps(3, steps=3, output_stem=Path(directory)/"three_steps")
            self.assertEqual([path.suffix for path in paths], [".png", ".pdf"])
            for path in paths:
                self.assertGreater(path.stat().st_size, 1000)

    def test_rejects_invalid_diagram_parameters(self):
        for kwargs in ({"number_of_qubits": 1}, {"steps": 0}, {"dt": 0}, {"dt": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                diagram.build_display_circuit(**kwargs)


if __name__ == "__main__":
    unittest.main()
