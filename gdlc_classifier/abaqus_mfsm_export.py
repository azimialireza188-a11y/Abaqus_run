"""High-throughput read-only Abaqus ODB -> portable FP64 U/UR shards.

Self-contained (no imports of sibling modules). It is called automatically by
gdlc_mode_decomposition.py (--export auto|always) as
    abaqus python abaqus_mfsm_export.py --odb <run>.odb --inp <run>.inp --output-dir <run>/modal_export_gdlc
and can still be run by hand with the same arguments.

Run with Abaqus Python. This exporter performs no classification and no solve.
The critical path uses Abaqus FieldBulkData when available, avoiding per-FieldValue
Python traversal. Compression is pipelined across all visible logical CPUs while
one ODB owner performs native bulk extraction. No standing RAM/VRAM reserve is
introduced; allocation recovery is triggered only by an actual MemoryError.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import numpy as np


# ---------------------------------------------------------------------------
# Self-contained INP reader and resource policy (no sibling-module imports, so the
# exporter runs from any folder, e.g. gdlc_classifier, and also on the GDLC
# reference models whose .inp contains *Coupling / assembly-level reference nodes).
# Only what the export needs is read: part nodes and elements, instances with their
# transforms, and the single *Buckle step. Every other keyword is ignored (it does
# not change the node/element topology that is checked against the ODB).
# ---------------------------------------------------------------------------
def _inp_fields(line):
    return [s.strip().strip('"\'') for s in line.split(',')]


def _inp_blocks(text):
    current=None
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('**'):continue
        if line.startswith('*'):
            if current is not None:yield current
            fields=_inp_fields(line[1:]);attrs={};flags=set()
            for token in fields[1:]:
                if '=' in token:
                    k,v=token.split('=',1);attrs[k.strip().upper()]=v.strip().strip('"\'')
                elif token:flags.add(token.upper())
            current=(fields[0].upper(),attrs,flags,[])
        elif current is not None:
            current[3].append(line)
    if current is not None:yield current


def _instance_points(nodes,lines):
    if len(lines)>2:raise ValueError('Unsupported instance transformation')
    labels=list(nodes);xyz=np.array([nodes[k] for k in labels],dtype=float)
    if lines:
        translation=np.array([float(x) for x in _inp_fields(lines[0])])
        if translation.shape!=(3,):raise ValueError('Instance translation needs 3 values')
        xyz=xyz+translation
    if len(lines)==2:
        values=np.array([float(x) for x in _inp_fields(lines[1])])
        if values.shape!=(7,):raise ValueError('Instance rotation needs 7 values')
        a=values[:3];axis=values[3:6]-a
        if np.linalg.norm(axis)==0:raise ValueError('Zero rotation axis')
        x,y,z=axis/np.linalg.norm(axis);theta=np.deg2rad(values[6])
        skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        rotation=np.eye(3)+np.sin(theta)*skew+(1-np.cos(theta))*(skew@skew)
        xyz=(xyz-a)@rotation.T+a
    if not np.all(np.isfinite(xyz)):raise ValueError('Nonfinite instance coordinates')
    return dict(zip(labels,xyz))


def read_input(path):
    raw=Path(path).read_bytes();text=raw.decode('utf-8-sig',errors='replace')
    parts={};instances={};steps=[];part=None;instance=None
    for key,attrs,flags,lines in _inp_blocks(text):
        if key in ('INCLUDE','PARAMETER') or 'INPUT' in attrs:
            raise ValueError('Expanded input required (no *INCLUDE / *PARAMETER / INPUT=): '+key)
        if key=='PART':
            part=attrs['NAME'].upper()
            if part in parts:raise ValueError('Duplicate part '+part)
            parts[part]=dict(nodes={},elements={})
        elif key=='END PART':part=None
        elif key=='INSTANCE':
            instance=attrs['NAME'].upper()
            if instance in instances:raise ValueError('Duplicate instance '+instance)
            if 'PART' not in attrs:raise ValueError('Instance without PART is not supported')
            instances[instance]=dict(part=attrs['PART'].upper(),transform=lines)
        elif key=='END INSTANCE':instance=None
        elif key=='NODE':
            if part is None:
                if instance is not None:raise ValueError('Instance-level nodes are not supported')
                continue            # assembly-level reference nodes: not part of the shell mesh
            if attrs.get('SYSTEM','R').upper()!='R':raise ValueError('Only rectangular NODE coordinates supported')
            for row in lines:
                f=_inp_fields(row);label=int(f[0]);values=[float(v) for v in f[1:4]]
                if len(values)!=3 or label in parts[part]['nodes']:raise ValueError('Invalid/duplicate 3D node')
                parts[part]['nodes'][label]=values
        elif key=='ELEMENT':
            if part is None:continue  # assembly-level connector elements are not shell elements
            etype=attrs.get('TYPE','').upper()
            for row in lines:
                f=[int(v) for v in _inp_fields(row) if v]
                if len(f)<2 or f[0] in parts[part]['elements']:raise ValueError('Invalid/duplicate element')
                parts[part]['elements'][f[0]]=(etype,tuple(f[1:]))
        elif key=='STEP':
            steps.append(dict(name=attrs.get('NAME','STEP-'+str(len(steps)+1)).upper(),attrs=attrs,flags=sorted(flags)))
        elif key=='BUCKLE':
            if not steps or not lines:raise ValueError('BUCKLE requires a step and a data line')
            steps[-1]['buckle_data']=_inp_fields(lines[0])
    nodes={};elements={}
    for name,spec in instances.items():
        if spec['part'] not in parts:raise ValueError('Unknown instance part '+spec['part'])
        p=parts[spec['part']]
        if not p['nodes'] or not p['elements']:raise ValueError('Empty part '+spec['part'])
        for label,xyz in _instance_points(p['nodes'],spec['transform']).items():nodes[(name,label)]=xyz
        for label,(etype,conn) in p['elements'].items():elements[(name,label)]=(etype,tuple((name,n) for n in conn))
    if not nodes:raise ValueError('No assembled shell nodes')
    if any(n not in nodes for _,conn in elements.values() for n in conn):raise ValueError('Element references missing node')
    return dict(nodes=nodes,elements=elements,instances=instances,steps=steps,
                input_sha256=hashlib.sha256(raw).hexdigest())


class _Policy:
    """All visible logical CPUs, no reserve (ODB access itself stays single-owner)."""
    def __init__(self):
        self.cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    def provenance(self):
        return dict(cpus=self.cpus,memory_reserve_bytes=0,gpu_reserve_bytes=0)


def detect_resources():
    return None


def resolve_policy(_inventory=None):
    return _Policy()


def _file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def _stable_snapshot(path, check_odb_lock=False):
    path=Path(path)
    lock=path.with_suffix('.lck')
    def state():
        if check_odb_lock and lock.exists():raise ValueError('ODB writer lock exists: '+str(lock))
        s=path.stat()
        return (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino)
    before=state();digest=_file_hash(path);after=state()
    if before!=after:raise ValueError('Source changed while hashing: '+str(path))
    return dict(sha256=digest,state=after)


def _attr(obj,name,default=None):
    try:value=getattr(obj,name)
    except Exception:return default
    try:return value() if callable(value) else value
    except Exception:return default


def _node_indexers(node_keys):
    """Build one reusable node-label -> output-row map for the whole export."""
    groups={}
    for row,(name,label) in enumerate(node_keys):
        groups.setdefault(str(name).upper(),[]).append((int(label),row))
    result={};kinds={}
    for name,pairs in groups.items():
        labels=np.asarray([p[0] for p in pairs],dtype=np.int64)
        rows=np.asarray([p[1] for p in pairs],dtype=np.int64)
        maximum=int(labels.max()) if len(labels) else -1
        # Dense lookup is substantially faster for ordinary Abaqus sequential labels.
        # Pathological sparse labels fall back to a dictionary instead of forcing a
        # huge artificial allocation.
        if maximum>=0 and maximum<=max(4096,16*len(labels)):
            dense=np.full(maximum+1,-1,dtype=np.int64);dense[labels]=rows
            result[name]=('dense',dense);kinds[name]='dense'
        else:
            result[name]=('dict',dict(zip(labels.tolist(),rows.tolist())));kinds[name]='dict'
    return result,kinds


def _positions(indexers,instance_name,labels):
    name=str(instance_name).upper()
    if name not in indexers:raise ValueError('ODB field contains unexpected instance: '+str(instance_name))
    kind,indexer=indexers[name]
    labels=np.asarray(labels,dtype=np.int64).ravel()
    if kind=='dense':
        if len(labels) and (labels.min()<0 or labels.max()>=len(indexer)):
            raise ValueError('ODB field node label outside mapped instance range')
        rows=indexer[labels]
        if np.any(rows<0):raise ValueError('ODB field contains unmapped node label')
        return rows
    try:return np.fromiter((indexer[int(label)] for label in labels),dtype=np.int64,count=len(labels))
    except KeyError as exc:raise ValueError('ODB field contains unmapped node label') from exc


def _block_array(block):
    precision=str(_attr(block,'precision',''))
    double='DOUBLE' in precision.upper()
    data_name='dataDouble' if double else 'data'
    data=_attr(block,data_name,None)
    if data is None:raise ValueError('Bulk field block has no '+data_name)
    array=np.asarray(data,dtype=np.float64)
    if array.ndim==1:
        width=int(_attr(block,'width',3) or 3)
        if width<3 or len(array)%width:raise ValueError('Invalid bulk vector width')
        array=array.reshape((-1,width))
    if array.ndim!=2 or array.shape[1]<3 or not np.all(np.isfinite(array[:,:3])):
        raise ValueError('Invalid/nonfinite bulk nodal vector data')
    orientation=int(_attr(block,'orientationWidth',0) or 0)
    local=_attr(block,'localCoordSystemDouble' if double else 'localCoordSystem',None)
    if orientation or (local is not None and np.size(local)):
        raise ValueError('Local-coordinate nodal output is not supported; use global U/UR')
    return array[:,:3],precision or 'UNKNOWN_PRECISION'


def _read_field_bulk(field,indexers,matrix,column):
    blocks=_attr(field,'bulkDataBlocks',None)
    if blocks is None:return False,set()
    try:blocks=list(blocks)
    except Exception:return False,set()
    if not blocks:return False,set()
    precision=set();filled=0
    for block in blocks:
        instance=_attr(block,'instance',None)
        labels=_attr(block,'nodeLabels',None)
        if labels is None:
            return False,set()
        if instance is None or str(_attr(instance,'name','')).upper() not in indexers:
            continue                  # assembly-level / reference-point nodes are not exported
        data,token=_block_array(block)
        rows=_positions(indexers,_attr(instance,'name',''),labels)
        if len(rows)!=len(data):raise ValueError('Bulk field labels/data length mismatch')
        if len(np.unique(rows))!=len(rows) or np.any(np.isfinite(matrix[rows,column])):
            raise ValueError('Duplicate bulk nodal field values')
        matrix[rows,column:column+3]=data
        precision.add(token);filled+=len(rows)
    return filled>0,precision


def _value_vector(value):
    precision=str(_attr(value,'precision',''))
    double='DOUBLE' in precision.upper()
    local_name='localCoordSystemDouble' if double else 'localCoordSystem'
    local=_attr(value,local_name,None)
    if local is not None:
        try:
            if np.size(local):raise ValueError('Local-coordinate nodal output is not supported; use global U/UR')
        except TypeError:
            raise ValueError('Local-coordinate nodal output is not supported; use global U/UR')
    data=_attr(value,'dataDouble' if double else 'data',None)
    vector=np.asarray(data,dtype=np.float64).ravel()
    if len(vector)<3 or not np.all(np.isfinite(vector[:3])):
        raise ValueError('Invalid/nonfinite nodal vector data')
    return vector[:3],precision or 'UNKNOWN_PRECISION'


def _read_field_values(field,indexers,matrix,column):
    precision=set()
    values=_attr(field,'values',None)
    if values is None:raise ValueError('ODB field has neither bulkDataBlocks nor values')
    for value in values:
        instance=_attr(value,'instance',None)
        if instance is None or str(_attr(instance,'name','')).upper() not in indexers:continue
        row=_positions(indexers,_attr(instance,'name',''),[int(_attr(value,'nodeLabel'))])[0]
        if np.isfinite(matrix[row,column]):raise ValueError('Duplicate nodal field value')
        vector,token=_value_vector(value)
        matrix[row,column:column+3]=vector;precision.add(token)
    return precision


def _read_mode(frame,indexers,node_count):
    """Return one node-major [U1,U2,U3,UR1,UR2,UR3] FP64 vector."""
    matrix=np.full((node_count,6),np.nan,dtype=np.float64)
    precision={};backends=[]
    for field_name,column in (('U',0),('UR',3)):
        if field_name not in frame.fieldOutputs:
            raise ValueError('Required U/UR field missing from ODB')
        field=frame.fieldOutputs[field_name]
        try:
            used_bulk,tokens=_read_field_bulk(field,indexers,matrix,column)
        except (ValueError, TypeError, RuntimeError, AttributeError):
            used_bulk,tokens=False,set()
        if used_bulk:
            backends.append('bulk')
        else:
            matrix[:,column:column+3]=np.nan
            tokens=_read_field_values(field,indexers,matrix,column);backends.append('values')
        precision[field_name]=sorted(tokens)
    if not np.all(np.isfinite(matrix)):raise ValueError('Incomplete mapped U/UR data')
    return matrix.reshape(-1),precision,backends


def _write_shard(path,vectors,modes,eigenvalues,compressed=True):
    started=time.perf_counter()
    writer=np.savez_compressed if compressed else np.savez
    writer(path,vectors=vectors,modes=modes,eigenvalues=eigenvalues)
    return dict(seconds=time.perf_counter()-started,bytes=os.path.getsize(path))


def export_modal_data(odb,model,output_dir,odb_sha256,modes_per_shard=8,source_guard=None,
                      compressed=True,progress_seconds=2.0):
    if type(modes_per_shard) is not int or modes_per_shard<1 or not odb_sha256:
        raise ValueError('Positive shard size and source ODB hash required')
    started=time.perf_counter();topology_started=time.perf_counter()
    names={str(name).upper():name for name in odb.rootAssembly.instances.keys()}
    missing=set(model['instances'])-set(names)
    extra=[n for n in set(names)-set(model['instances']) if len(odb.rootAssembly.instances[names[n]].elements)]
    if missing or extra:raise ValueError('ODB/INP instances differ: missing=%s extra=%s'%(sorted(missing),sorted(extra)))
    coordinates=[];keys=[];node_keys=[]
    for name in model['instances']:
        inst=odb.rootAssembly.instances[names[name]]
        actual={int(n.label):np.asarray(n.coordinates,dtype=float) for n in inst.nodes}
        expected={label:xyz for (part,label),xyz in model['nodes'].items() if part==name}
        if set(actual)!=set(expected):raise ValueError('ODB/INP node labels differ')
        for label,xyz in expected.items():
            if not np.allclose(actual[label],xyz,rtol=0,atol=max(1e-5,1e-6*np.max(np.abs(xyz)))):
                raise ValueError('ODB/INP coordinates differ')
            coordinates.append(actual[label]);node_keys.append((names[name],label))
            keys.extend((names[name],label,dof) for dof in range(1,7))
        actual_elements={int(e.label):tuple(int(n) for n in e.connectivity) for e in inst.elements}
        expected_elements={label:tuple(n[1] for n in conn) for (part,label),(etype,conn) in model['elements'].items() if part==name}
        if actual_elements!=expected_elements:
            raise ValueError('ODB/INP element connectivity differs for instance '+name)
    indexers,indexer_kinds=_node_indexers(node_keys)
    if len(model['steps'])!=1 or 'buckle_data' not in model['steps'][0]:
        raise ValueError('Exactly one source BUCKLE step required')
    step_name=model['steps'][0]['name']
    matched=[name for name in odb.steps.keys() if str(name).upper()==step_name]
    if len(matched)!=1:raise ValueError('ODB/INP buckling step mismatch')
    frames=[f for f in odb.steps[matched[0]].frames if int(getattr(f,'mode',0))>0]
    ids=[int(f.mode) for f in frames]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Empty/duplicate buckling mode IDs')
    topology_seconds=time.perf_counter()-topology_started
    root=Path(output_dir)
    try:root.mkdir(parents=True,exist_ok=False)
    except FileExistsError as exc:raise ValueError('Use a new export directory') from exc
    policy=resolve_policy(detect_resources())
    report=dict(kind='PORTABLE_RAW_MODAL_DATA_ONLY',format='npz',exporter='abaqus_mfsm_export.py',
        source_odb=str(_attr(odb,'path','')),source_inp=model.get('input_path'),source_odb_sha256=odb_sha256,
        source_inp_sha256=model['input_sha256'],step=str(matched[0]),mode_count=len(frames),
        node_count=len(node_keys),raw_dof_count=len(keys),modes=[],shards=[],
        scientifically_eligible=False,default_classifier_activation=False,
        source_stability_verified=False,
        contact_state='UNKNOWN_REQUIRES_BASE_STATE_EVIDENCE',resources=policy.provenance(),
        modes_per_shard=modes_per_shard,processing_precision='FP64_PRESERVING_SOURCE_DATA',
        extraction_backend_counts=dict(bulk_fields=0,value_fields=0),
        node_indexer_kinds=indexer_kinds,compression=('ZLIB' if compressed else 'STORE'),
        execution_policy=dict(odb_owner_threads=1,compression_workers=policy.cpus,
            memory_reserve_bytes=0,gpu_reserve_bytes=0,
            gpu_usage='NOT_APPLICABLE_TO_NATIVE_ODB_IO_OR_NPZ_COMPRESSION'),
        timings=dict(topology_seconds=topology_seconds,odb_extraction_seconds=0.,
                     compression_task_seconds_total=0.,compression_task_seconds_max=0.),
        limitations=['No S4R/contact tangent or mechanical basis in this archive.',
                     'FP64 storage does not recover accuracy lost in source output.',
                     'Abaqus ODB access is owned by one process; native bulk blocks minimize Python overhead.'])
    print('[MFSM EXPORT] modes=%d nodes=%d rawDOF=%d CPUs=%d backend=BULK_PREFERRED compression=%s' %
          (len(frames),len(node_keys),len(keys),policy.cpus,report['compression']))
    sys.stdout.flush()
    with tempfile.TemporaryDirectory(prefix='modal-stage-',dir=root) as stage:
        np.savez_compressed(Path(stage,'raw_dof_map.npz'),instances=[k[0] for k in keys],
            labels=[k[1] for k in keys],dofs=[k[2] for k in keys],node_instances=[k[0] for k in node_keys],
            node_labels=[k[1] for k in node_keys],node_coordinates=np.asarray(coordinates),
            metadata=json.dumps(dict(source_inp_sha256=model['input_sha256'],source_odb_sha256=odb_sha256)))
        pending=set();compression_stats=[]
        def drain(all_tasks=False):
            nonlocal pending
            done,pending=wait(pending,return_when='ALL_COMPLETED' if all_tasks else FIRST_COMPLETED)
            for task in done:compression_stats.append(task.result())
        extracted=0;extract_started=time.perf_counter();last_progress=extract_started
        with ThreadPoolExecutor(max_workers=policy.cpus) as pool:
            for offset in range(0,len(frames),modes_per_shard):
                if len(pending)>=policy.cpus:drain()
                selected=frames[offset:offset+modes_per_shard]
                while True:
                    try:vectors=np.empty((len(keys),len(selected)),dtype=np.float64);break
                    except MemoryError:
                        if not pending:raise
                        drain()
                shard_modes=[];shard_eigenvalues=[]
                for j,frame in enumerate(selected):
                    match=re.search(r'EigenValue\s*=\s*([-+0-9.EeDd]+)',frame.description,re.I)
                    if not match:raise ValueError('Missing eigenvalue in mode '+str(frame.mode))
                    eigenvalue=float(match.group(1).replace('D','E').replace('d','e'))
                    if not np.isfinite(eigenvalue):raise ValueError('Nonfinite eigenvalue')
                    one_started=time.perf_counter()
                    vector,precision,backends=_read_mode(frame,indexers,len(node_keys))
                    report['timings']['odb_extraction_seconds']+=time.perf_counter()-one_started
                    vectors[:,j]=vector
                    for backend in backends:
                        report['extraction_backend_counts'][backend+'_fields']+=1
                    mode=int(frame.mode);shard_modes.append(mode);shard_eigenvalues.append(eigenvalue)
                    report['modes'].append(dict(mode=mode,eigenvalue=eigenvalue,
                        fields=sorted(frame.fieldOutputs.keys()),source_precision=precision))
                    extracted+=1;now=time.perf_counter()
                    if now-last_progress>=progress_seconds or extracted==len(frames):
                        elapsed=now-extract_started;rate=extracted/max(elapsed,1e-12)
                        eta=(len(frames)-extracted)/max(rate,1e-12)
                        print('[MFSM EXPORT] extracted %d/%d (%.1f%%) %.2f mode/s ETA %.1fs bulk=%d fallback=%d pending_zip=%d' %
                              (extracted,len(frames),100.*extracted/len(frames),rate,eta,
                               report['extraction_backend_counts']['bulk_fields'],
                               report['extraction_backend_counts']['value_fields'],len(pending)))
                        sys.stdout.flush();last_progress=now
                filename='modes_%04d.npz'%(len(report['shards'])+1)
                report['shards'].append(filename)
                pending.add(pool.submit(_write_shard,Path(stage,filename),vectors,
                    np.asarray(shard_modes,dtype=np.int64),np.asarray(shard_eigenvalues,dtype=float),compressed))
            if pending:drain(True)
        if compression_stats:
            report['timings']['compression_task_seconds_total']=sum(x['seconds'] for x in compression_stats)
            report['timings']['compression_task_seconds_max']=max(x['seconds'] for x in compression_stats)
            report['compressed_bytes']=sum(x['bytes'] for x in compression_stats)
        artifacts=['raw_dof_map.npz']+report['shards']
        hash_started=time.perf_counter()
        report['artifact_sha256']={name:_file_hash(Path(stage,name)) for name in artifacts}
        report['timings']['artifact_hash_seconds']=time.perf_counter()-hash_started
        if source_guard is not None:
            guard_started=time.perf_counter();source_guard()
            report['timings']['source_guard_seconds']=time.perf_counter()-guard_started
            report['source_stability_verified']=True
        report['elapsed_seconds']=time.perf_counter()-started
        Path(stage,'modal_export.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        for name in artifacts+['modal_export.json']:os.replace(Path(stage,name),root/name)
    print('[MFSM EXPORT] COMPLETE elapsed=%.2fs extract=%.2fs compression_task_total=%.2fs bulk_fields=%d fallback_fields=%d' %
          (report['elapsed_seconds'],report['timings']['odb_extraction_seconds'],
           report['timings']['compression_task_seconds_total'],
           report['extraction_backend_counts']['bulk_fields'],
           report['extraction_backend_counts']['value_fields']))
    sys.stdout.flush()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--odb',required=True);p.add_argument('--inp',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--modes-per-shard',type=int,default=8)
    p.add_argument('--compression',choices=('zlib','store'),default='zlib',
                   help='zlib: smaller transfer files with parallel CPU compression; store: fastest write/larger files')
    p.add_argument('--progress-seconds',type=float,default=2.0)
    args=p.parse_args()
    initial_odb=_stable_snapshot(args.odb,True);initial_inp=_stable_snapshot(args.inp)
    model=read_input(args.inp);model['input_path']=os.path.abspath(args.inp)
    if model['input_sha256']!=initial_inp['sha256']:raise ValueError('INP source changed during parsing')
    def guard():
        if _stable_snapshot(args.odb,True)!=initial_odb:raise ValueError('ODB source changed during export')
        if _stable_snapshot(args.inp)!=initial_inp:raise ValueError('INP source changed during export')
    from odbAccess import openOdb
    odb=openOdb(path=os.path.abspath(args.odb),readOnly=True)
    try:r=export_modal_data(odb,model,args.output_dir,initial_odb['sha256'],args.modes_per_shard,
                            source_guard=guard,compressed=(args.compression=='zlib'),
                            progress_seconds=max(.1,args.progress_seconds))
    finally:odb.close()
    print(json.dumps(dict(mode_count=r['mode_count'],shards=r['shards'],output_dir=args.output_dir,
                         elapsed_seconds=r['elapsed_seconds'],
                         extraction_backend_counts=r['extraction_backend_counts'],
                         scientifically_eligible=False)))


if __name__=='__main__':main()
