"""Same native CZ twirling/folding and controls as the manuscript run."""
from __future__ import annotations
import gzip
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit,qpy,transpile
from qiskit.circuit.library import XXPlusYYGate
from FinalModel.Model3Experiment import model
from Model3.Experiment5 import circuit_depth as old_circuits,mitigation as original


def body(case,choices,control=None):
    n=case['N'];p=case['physics'];qc=QuantumCircuit(n+2)
    populations=np.diag(model.study.initial(n,p['preparation'])) if control is None else {
        'all_zero':np.zeros(n),'all_one':np.ones(n),'initial_mixture':np.full(n,.5)}[control]
    old_circuits.add_preparation(qc,list(range(n)),populations)
    if control is None:
        for dt,choice in zip(model.store.steps(case),choices):
            angle=4*p['prefactor']*p['J']*dt
            for parity in (0,1):
                for j in range(parity,n-1,2):qc.append(XXPlusYYGate(angle),[j,j+1])
            side=int(choice)//2;bath=n+side
            if choice%2==0:qc.x(bath)
            qc.append(XXPlusYYGate(2*np.sqrt(2*p['gamma']*dt)),[n-1 if side else 0,bath])
            qc.reset(bath)
    return qc


def transform_seed(cfg,row):
    index=row.get('trajectory',{'all_zero':0,'all_one':1,'initial_mixture':2}.get(row.get('control'),0))
    return int(np.random.SeedSequence([cfg['mitigation']['seed'],index,row['variant'],
               0 if row['kind']=='trajectory' else 1]).generate_state(1)[0])


def assert_supported(qc,target):
    for item in qc.data:
        if item.operation.name=='barrier':continue
        args=tuple(qc.find_bit(q).index for q in item.qubits)
        if not target.instruction_supported(operation_name=item.operation.name,qargs=args):
            raise ValueError(f'Unsupported native operation {item.operation.name}{args}')


def compile_group(source,data,cfg,rows,target,compiler_methods):
    case=source['case'];n=case['N']
    layout,path=model.store.chain_layout(target,n,supplied=cfg['hardware']['physical_path'])
    templates={};bodies={};pubs=[];out=[]
    for row in rows:
        key=(row['kind'],row.get('trajectory',row.get('control')),row['basis'])
        if key not in templates:
            base_key=key[:2]
            if base_key not in bodies:
                qc=body(case,data['choices'][row.get('trajectory',0)],row.get('control'))
                native=transpile(qc,target=target,initial_layout=layout,routing_method='none',
                    optimization_level=cfg['hardware']['optimization_level'],seed_transpiler=cfg['hardware']['transpiler_seed'],
                    num_processes=1,**compiler_methods)
                if native.layout.final_index_layout(filter_ancillas=True)!=layout:raise ValueError('Fixed physical mapping changed')
                bodies[base_key]=native
            templates[key]=old_circuits.append_readout(bodies[base_key],layout[:n],row['basis'])
        native=templates[key];cz=int(native.count_ops().get('cz',0))
        if row['kind']=='trajectory' and native.count_ops().get('reset',0)!=case['steps']:raise ValueError('Missing bath resets')
        factor=cfg['mitigation']['fold_factors'][row.get('factor_index',0)]
        folds=original.fold_count(factor,cz) if row['kind']=='trajectory' and row['arm']=='twirled' else 0
        qc=native.copy() if row['arm']=='raw' else original.transformed(native,seed=transform_seed(cfg,row),folds=folds,measurement_mask=row['mask'])
        qc.metadata={**(qc.metadata or {}),'case_id':row['case_id']}
        assert_supported(qc,target)
        out.append(dict(row,base_cz=cz,fold_count=folds,actual_factor=1+2*folds/cz if cz else 1.,
                        metrics=old_circuits.metrics(qc),physical_sites=layout[:n]))
        pubs.append(qc)
    return pubs,out,path


def compile_batch(folder,source,data,batch,target,metadata,*,preview_paths=None):
    cfg=source['configuration'];folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    selected=dict(batch)
    if preview_paths is not None:selected['indices']=batch['indices'][:preview_paths]
    expected=model.records(cfg,selected)
    stem=f"batch_{batch['index']:05d}"
    qpy_file=folder/f'{stem}.qpy.gz';record_file=folder/f'{stem}.json'
    if record_file.exists():
        record=model.store.read_sealed(record_file)
        if (record['source_sha256']!=source['sha256'] or record['target_sha256']!=metadata['target_sha256']
                or record['plan_sha256']!=model.store.digest(expected)
                or model.store.file_digest(qpy_file)!=record['qpy_sha256']):raise ValueError('Compiled batch provenance changed')
        with gzip.open(qpy_file,'rb') as stream:pubs=qpy.load(stream)
        if len(pubs)!=len(expected):raise ValueError('QPY circuit count mismatch')
        for qc,row in zip(pubs,record['circuits']):
            if qc.metadata.get('case_id')!=row['case_id'] or old_circuits.metrics(qc)!=row['metrics']:raise ValueError('QPY contents changed')
            assert_supported(qc,target)
        return pubs,record['circuits']
    # Compile one matched path group at a time; controls and readout bases
    # keep the same physical sites. Release each group's template cache.
    pubs=[];records=[];pos=0
    while pos<len(expected):
        key=expected[pos].get('trajectory',expected[pos].get('control'));kind=expected[pos]['kind'];stop=pos+1
        while stop<len(expected) and expected[stop]['kind']==kind and expected[stop].get('trajectory',expected[stop].get('control'))==key:stop+=1
        group,rows,path=compile_group(source,data,cfg,expected[pos:stop],target,metadata.get('compiler_methods',{}))
        pubs.extend(group);records.extend(rows);pos=stop
        if len(records)%390==0 or pos==len(expected):print(f"Compiled {len(records)}/{len(expected)} PUBs in batch {batch['index']}",flush=True)
    with gzip.open(qpy_file,'wb',compresslevel=1) as stream:qpy.dump(pubs,stream)
    record=model.store.sealed(dict(source_sha256=source['sha256'],target_sha256=metadata['target_sha256'],
        plan_sha256=model.store.digest(expected),qpy_sha256=model.store.file_digest(qpy_file),physical_path=path,circuits=records))
    model.store.atomic_json(record_file,record)
    return pubs,records
