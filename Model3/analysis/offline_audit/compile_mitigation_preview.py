"""Compile one mitigation trajectory on a saved target; no IBM access."""
import hashlib
import json
from pathlib import Path

from qiskit import transpile
from qiskit.circuit import Measure,Parameter,Reset
from qiskit.circuit.library import CZGate,RZGate,SXGate,XGate
from qiskit.transpiler import InstructionProperties,Target

from Model3.Experiment5 import circuit_depth,run_mitigation
from Model3.Experiment5.run_schedule_comparison import load_inputs,DEFAULT_INPUT

ROOT=Path(__file__).resolve().parents[3]


def main():
    path=ROOT/'Model3/Experiment5/hardware_runs/ibm_phoenix_mapping_comparison_20261026T050356_310922Z/backend_snapshot.json'
    snapshot=json.loads(path.read_bytes());target=Target(num_qubits=120)
    for name,op in [('cz',CZGate()),('x',XGate()),('sx',SXGate()),('measure',Measure()),('reset',Reset())]:
        target.add_instruction(op,{tuple(p['qubits']):InstructionProperties(duration=p['duration_seconds'],error=p['error'])
            for p in snapshot['instructions'] if p['name']==name})
    target.add_instruction(RZGate(Parameter('phi')),{(int(q),):InstructionProperties(duration=0,error=0) for q in snapshot['qubits']})
    sources,raw,_,_=load_inputs(DEFAULT_INPUT);source=sources['candidate'];options=run_mitigation.parse_arguments([])
    cfg={k:getattr(options,k) for k in ('physical_path','trajectory_count','shots','randomizations','control_shots','fold_factors','seed','batch_size')}
    cfg['steps']=len(source['schedule']['step_dts']);records,_=run_mitigation.plan(cfg)
    body=circuit_depth.build_body(source,0);p=cfg['physical_path'];layout=p[1:-1]+[p[0],p[-1]]
    native=transpile(body,target=target,initial_layout=layout,seed_transpiler=11,optimization_level=3)
    assert native.layout.final_index_layout(filter_ancillas=True)==layout
    native=circuit_depth.append_readout(native,layout[:10],'z');rows=[]
    backend=type('SavedTarget',(),dict(target=target))()
    for record in records:
        if record['kind']!='trajectory' or record['trajectory']!=0:continue
        circuit=run_mitigation.make_variant(native,cfg,record);run_mitigation.assert_supported(circuit,backend)
        rows.append(dict(arm=record['arm'],factor=record['actual_factor'],variant=record['randomization'],metrics=circuit_depth.metrics(circuit)))
    result=dict(scope='One representative saved trajectory compiled on a reconstructed saved CZ target. No fresh calibration, IBM connection, simulator or execution. Not a binary reproduction of original QPY or a measured duration.',
        source_sha256=hashlib.sha256(raw['candidate']).hexdigest(),snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        physical_path=p,trajectory=0,records=rows)
    output=ROOT/'Model3/Experiment5/inputs/mitigation_compilation_preview.json';output.write_text(json.dumps(result,indent=2)+'\n')
    for arm,factor in [('raw',1.),('twirled',1.),('twirled',1.25),('twirled',1.5)]:
        selected=[r['metrics'] for r in rows if r['arm']==arm and r['factor']==factor]
        print(arm,factor,'depth',min(r['depth'] for r in selected),max(r['depth'] for r in selected),
            'CZ',selected[0]['operation_counts']['cz'],'X range',min(r['operation_counts'].get('x',0) for r in selected),max(r['operation_counts'].get('x',0) for r in selected),flush=True)
    return result


if __name__=='__main__':main()
