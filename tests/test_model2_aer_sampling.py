"""Exact bath reduction, Aer samples, register decoding and shot-budget bounds."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import ClassicalRegister,QuantumCircuit
from qiskit.quantum_info import DensityMatrix,Kraus

from Model2.Experiment5 import aer_sampling,run_hardware,input_data


def exact_density(circuit):
    density=DensityMatrix.from_label("0"*circuit.num_qubits)
    for instruction in circuit.data:
        args=[circuit.find_bit(bit).index for bit in instruction.qubits]
        op=instruction.operation
        if op.name=="measure":
            density=density.evolve(Kraus([np.diag([1,0]),np.diag([0,1])]),args)
        elif op.name=="reset":
            density=density.reset(args)
        elif op.name=="kraus":
            density=density.evolve(Kraus(op.params),args)
        else:
            density=density.evolve(op,args)
    return density


class AerSamplingTests(unittest.TestCase):
    def test_reduced_and_full_gate_lowering_are_exact_for_all_boundary_branches(self):
        from qiskit.quantum_info import partial_trace
        spec=input_data.load_spec()
        source={"configuration":{"physics":{**spec["physics"],"n_qubits":3},"trajectories":1,"order":"jump-first"},
                "preparation":{"populations":[.5]*3},
                "schedule":{"step_dts":[4.207281841881579,4.,3.9,3.760760459292396],
                            "jump_choices":[[0,1,2,3]],"jump_probabilities":[.125,.375,.125,.375]}}
        for length in range(1,5):
            body=run_hardware.build_body(source,0,steps=length)
            full=aer_sampling.aer_body(body,3,True)
            reduced=aer_sampling.aer_body(body,3,False)
            expected=partial_trace(exact_density(body),[3,4])
            np.testing.assert_allclose(partial_trace(exact_density(full),[3,4]).data,expected.data,atol=3e-14)
            np.testing.assert_allclose(exact_density(reduced).data,expected.data,atol=3e-14)
            self.assertEqual(full.count_ops()["reset"],length)
            self.assertEqual(reduced.count_ops()["kraus"],3+length)
            self.assertFalse(any(inst.operation.name.startswith("save_") for inst in reduced.data))

    def test_outcome_decoder_ignores_preparation_register_and_uses_actual_offsets(self):
        circuit=QuantumCircuit(3)
        circuit.add_register(ClassicalRegister(3,"preparation"))
        circuit.add_register(ClassicalRegister(3,"outcome"))
        np.testing.assert_array_equal(aer_sampling.packed_memory(["101 011","101 111"],circuit,3),[5,5])

    def test_all_basis_estimator_preserves_current_parity(self):
        samples=np.zeros((2,3,2),dtype=np.uint16)
        samples[:,0]=16
        samples[:,1]=1
        samples[:,2]=4
        values=aer_sampling.estimate(samples)
        self.assertTrue(np.all(values[:,4]==1))
        np.testing.assert_allclose(values[:,10:13],[[.5,.5,-.5]]*2)
        np.testing.assert_allclose(values[:,13:],0.)

    def synthetic(self,finite,reference):
        return {"final_covariances_real_gauge":[(finite*np.eye(10)).tolist()]*16,
                "mean_channel_observables":{"populations":[[[reference]*10]],"excitation_flows":[[[0.]*9]]}}

    def test_more_shots_cannot_certify_a_fixed_ensemble_outside_sampling_budget(self):
        rows=aer_sampling.planning_rows(self.synthetic(.27,.25),[16],[8,1000000])
        self.assertTrue(all(not row["sufficient"] for row in rows))
        self.assertEqual([row["simultaneous_failure_probability_bound"] for row in rows],[1.,1.])

    def test_shot_bound_tightens_and_includes_hardware_control_workload(self):
        rows=aer_sampling.planning_rows(self.synthetic(.25,.25),[16],[8,1000000])
        self.assertGreater(rows[0]["simultaneous_failure_probability_bound"],rows[1]["simultaneous_failure_probability_bound"])
        self.assertTrue(rows[1]["sufficient"])
        self.assertEqual(rows[0]["hardware_shots_including_controls"],3*16*8+3*2048)

    def test_small_aer_run_and_verified_resume_are_offline(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(run_hardware.runtime,"make_service",side_effect=AssertionError("no IBM account")):
            args=["--paths","2","--shots","2","--threads","1","--chunk","2","--output-directory",folder]
            first=aer_sampling.main(args)
            self.assertTrue(first["complete"])
            with patch.object(aer_sampling.AerSimulator,"run",side_effect=AssertionError("cached samples must be reused")):
                second=aer_sampling.main(args+["--resume"])
            self.assertEqual(first,second)
            sample=next(Path(folder).glob("*/*.npz"))
            with sample.open("ab") as output: output.write(b" ")
            with self.assertRaisesRegex(ValueError,"samples changed"):
                aer_sampling.main(args+["--resume"])


if __name__=="__main__":
    unittest.main()
