# -*- coding: utf-8 -*-
"""Build only: STEP4 imperfect CAE -> STEP5 Riks CAE + one INP per selected model.

Abaqus 2024:
  abaqus cae noGUI=abaqus_step5_gmnia.py -- --source-cae "Step4_imperfections.cae"
    --output-dir "new_step5_folder" --fy 350

fy is REQUIRED (MPa). Elastic properties, thickness, mesh, imperfections, initial
BCs and general contact are inherited. Plasticity is elastic-perfectly-plastic,
with no damage, rate dependence, residual stress or bolt failure added.
All STEP4_* models are selected unless --models explicitly lists a subset.
The compatible source is the Z-axis four-piece model from the preceding scripts.

Rigid BEAM MPC bolts are replaced by BEAM assembled connectors at identical
nodes. These are force-output-capable rigid connectors, NOT MPC-type sections,
and NOT point-based fasteners with a new coupling footprint. Connector local
axis 1 follows node A -> B; local axis 2 is along the column projected normal to
axis 1. Forces CTF1..3 and moments CTM1..3 are recorded for every bolt.

Symmetric end tractions are retained, scaled to --reference-stress (default fy).
LPF=1 therefore means sigma_ref, not 1 MPa unless explicitly requested.
P=LPF*reference_force_N_per_end; never add the two opposing end forces.
End-area weights in each Model description define work-conjugate shortening:
delta = weighted_mean(U3 at zmin) - weighted_mean(U3 at zmax).
The midspan axial anchor reaction is NOT the column load.

Default stop: positive U3 at one bottom-end node reaches 0.01*L. This is a
single-node displacement limit, NOT total shortening. Max increments=1000;
initial/min/max arc increments=0.01/1e-8/0.05, total arc scale=1. These are pilot
settings, not a convergence guarantee. No equilibrium tolerances are loosened.
Confirm a peak and a sufficient descending branch after the eventual solve.

Only CAE/INP deliverables are placed in --output-dir (must be new or empty).
Jobs are created for later manual use; this script NEVER submits them. Abaqus
may create its own startup replay files in the launch directory. Source CAE is
not saved over. Do not remesh the imperfect models. Units: N, mm, MPa.
"""
import argparse
import builtins as bi
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile


def parse_arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        argv = argv[argv.index('--')+1:]
    elif '-cae' in argv:
        clean, i = [], 0
        while i < len(argv):
            if argv[i] == '-cae':
                i += 1
            elif argv[i] in ('-noGUI', '-tmpdir', '-lmlog'):
                i += 2
            else:
                clean.append(argv[i]); i += 1
        argv = clean
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-cae', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--fy', type=float, required=True)
    p.add_argument('--models', nargs='+')
    p.add_argument('--reference-stress', type=float)
    p.add_argument('--initial-arc', type=float, default=.01)
    p.add_argument('--min-arc', type=float, default=1e-8)
    p.add_argument('--max-arc', type=float, default=.05)
    p.add_argument('--max-increments', type=int, default=1000)
    p.add_argument('--max-end-displacement-mm', type=float)
    p.add_argument('--field-frequency', type=int, default=1)
    p.add_argument('--cpus', type=int, default=8)
    args = p.parse_args(argv)
    if args.reference_stress is None:
        args.reference_stress = args.fy
    for name in ('fy', 'reference_stress', 'initial_arc', 'min_arc', 'max_arc', 'max_end_displacement_mm'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error('--%s must be positive and finite' % name.replace('_', '-'))
    if not args.min_arc <= args.initial_arc <= args.max_arc:
        p.error('Require min-arc <= initial-arc <= max-arc')
    if min(args.cpus, args.max_increments, args.field_frequency) < 1:
        p.error('cpus, max-increments and field-frequency must be positive')
    args.source_cae = os.path.abspath(os.path.expanduser(args.source_cae))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    return args


def end_weights(coordinates, connectivity, thickness, zmin, zmax):
    """Tributary section areas for linear S4R edges on each column end."""
    result = ({}, {})
    tol = max(1e-6, (zmax-zmin)*1e-8)
    for labels in connectivity:
        for a, b in zip(labels, labels[1:]+labels[:1]):
            pa, pb = coordinates[a], coordinates[b]
            for side, z in enumerate((zmin, zmax)):
                if abs(pa[2]-z) <= tol and abs(pb[2]-z) <= tol:
                    weight = .5*thickness*math.sqrt(bi.sum((pa[j]-pb[j])**2 for j in range(3)))
                    result[side][a] = result[side].get(a, 0.)+weight
                    result[side][b] = result[side].get(b, 0.)+weight
    return result


def mesh_digest(model, names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode('utf-8'))
        instance = model.rootAssembly.instances[name]
        for node in instance.nodes:
            digest.update(struct.pack('<q3d', node.label, *node.coordinates))
        for element in instance.elements:
            labels = (element.label,)+tuple(n.label for n in element.getNodes())
            digest.update(struct.pack('<%dq' % len(labels), *labels))
    return digest.hexdigest()


def replace_bolts(model):
    from abaqusConstants import BEAM, BEAM_MPC, OFF, IMPRINT, CARTESIAN
    import numpy as np
    assembly = model.rootAssembly
    if len(assembly.edges):
        raise ValueError('Unexpected existing assembly wires; use the STEP4 source')
    bolt_names = sorted(name for name in model.constraints.keys() if name.startswith('BOLT_'))
    if not bolt_names:
        raise ValueError('No BOLT_* MPC connections found')
    pairs, metadata = [], {}
    for name in bolt_names:
        constraint = model.constraints[name]
        if getattr(constraint, 'mpcType', None) != BEAM_MPC:
            raise ValueError('Expected a BEAM MPC for '+name)
        a, b = assembly.sets[name+'_A'].nodes, assembly.sets[name+'_B'].nodes
        if len(a) != 1 or len(b) != 1:
            raise ValueError('Each bolt endpoint must contain exactly one node: '+name)
        if constraint.controlPoint[0] != name+'_A' or constraint.surface[0] != name+'_B':
            raise ValueError('Bolt endpoint sets do not match the MPC: '+name)
        pairs.append((a[0], b[0]))
        metadata[name] = dict(node_a=[a[0].instanceName, a[0].label],
                              node_b=[b[0].instanceName, b[0].label])
    print('  Creating %d rigid bolt connectors...' % len(pairs)); sys.stdout.flush()
    assembly.WirePolyLine(points=tuple(pairs), mergeType=IMPRINT, meshable=OFF)
    if len(assembly.edges) != len(pairs):
        raise RuntimeError('Connector wire count differs from original bolt count')
    model.ConnectorSection(name='STEP5_RIGID_BOLT', assembledType=BEAM)
    assembly.Set(name='STEP5_BOLTS', edges=assembly.edges[:])
    assembly.SectionAssignment(region=assembly.sets['STEP5_BOLTS'], sectionName='STEP5_RIGID_BOLT')
    for name, (a, b) in zip(bolt_names, pairs):
        pa, pb = np.asarray(a.coordinates), np.asarray(b.coordinates)
        midpoint = tuple(.5*(pa+pb))
        edges = assembly.edges.findAt((midpoint,))
        region = assembly.Set(name=name+'_CONNECTOR', edges=edges)
        axis = pb-pa
        if np.linalg.norm(axis) <= 1e-9:
            raise ValueError('Coincident bolt endpoints: '+name)
        axis /= np.linalg.norm(axis)
        guide = np.array([0., 0., 1.])
        if abs(np.dot(axis, guide)) > .99:
            guide = np.array([0., 1., 0.])
        datum = assembly.DatumCsysByThreePoints(name=name+'_CSYS', coordSysType=CARTESIAN,
            origin=tuple(pa), point1=tuple(pb), point2=tuple(pa+guide))
        assembly.ConnectorOrientation(region=region, localCsys1=assembly.datums[datum.id])
        del model.constraints[name]
    return metadata


def prepare_model(model, args):
    from abaqusConstants import ON, OFF, UNIFORM, GENERAL
    assembly = model.rootAssembly
    names = sorted(assembly.instances.keys())
    if len(names) != 4:
        raise ValueError('Expected exactly four shell instances')
    if any(str(e.type) != 'S4R' for name in names for e in assembly.instances[name].elements):
        raise ValueError('This builder supports the S4R mesh from the preceding scripts')
    if set(model.steps.keys()) != {'Initial', 'Buckle'}:
        raise ValueError('Expected only Initial and Buckle in source')
    if len(model.interactions) == 0:
        raise ValueError('Source contact interaction is missing')
    fingerprint = mesh_digest(model, names)
    prior_description = model.description
    coordinates = [n.coordinates for name in names for n in assembly.instances[name].nodes]
    zmin, zmax = min(p[2] for p in coordinates), max(p[2] for p in coordinates)
    length = zmax-zmin
    if length <= 0:
        raise ValueError('Expected a Z-axis column')
    weights, end_labels, end_area = [[], []], [[], []], [0., 0.]
    expected_loads, loads, materials, integration_counts = set(), [], set(), set()
    for name in names:
        inst = assembly.instances[name]
        assignments = model.parts[inst.partName].sectionAssignments
        section_names = set(a.sectionName for a in assignments)
        if len(section_names) != 1:
            raise ValueError('Expected one uniform shell section for '+name)
        section = model.sections[next(iter(section_names))]
        thickness = float(section.thickness)
        integration_counts.add(int(section.numIntPts))
        materials.add(section.material)
        xyz = {n.label: n.coordinates for n in inst.nodes}
        topology = [tuple(n.label for n in e.getNodes()) for e in inst.elements]
        tributary = end_weights(xyz, topology, thickness, zmin, zmax)
        for side, suffix, sign in ((0, '0', 1.), (1, 'L', -1.)):
            end = 'END_%s_%s' % (suffix, name)
            load_name = 'COMP_'+end
            expected_loads.add(load_name)
            old_load = model.loads[load_name]
            if old_load.region[0] != end+'_S':
                raise ValueError('Unexpected load surface: '+load_name)
            geometric_area = thickness*bi.sum(e.getSize(printResults=False) for e in assembly.sets[end].edges)
            meshed_area = bi.sum(tributary[side].values())
            if min(geometric_area, meshed_area) <= 0:
                raise ValueError('Missing loaded end geometry/mesh; check the column axis')
            # S4R edge loads act on the discretized edges. Curved geometry may
            # have a slightly larger arc length than the mesh's straight chords.
            end_area[side] += meshed_area
            end_labels[side].append((name, tuple(sorted(tributary[side]))))
            weights[side].extend([name, label, area] for label, area in sorted(tributary[side].items()))
            loads.append((load_name, end+'_S', thickness*args.reference_stress, sign))
    if set(model.loads.keys()) != expected_loads:
        raise ValueError('Unexpected additional/missing loads; refusing to silently remove them')
    if not math.isclose(end_area[0], end_area[1], rel_tol=1e-6):
        raise ValueError('Opposite loaded end areas differ')
    if len(integration_counts) != 1 or min(integration_counts) < 1:
        raise ValueError('Shell sections must have the same number of thickness integration points')
    for material_name in materials:
        material = model.materials[material_name]
        if not hasattr(material, 'elastic'):
            raise ValueError('Source elastic properties missing: '+material_name)
        if hasattr(material, 'plastic'):
            raise ValueError('Source already has plasticity; use the elastic STEP4 CAE')
        material.Plastic(table=((args.fy, 0.),))
    for repository in (model.loads, model.fieldOutputRequests, model.historyOutputRequests):
        for key in list(repository.keys()):
            del repository[key]
    del model.steps['Buckle']
    bolts = replace_bolts(model)
    print('  Defining Riks loading and response outputs...'); sys.stdout.flush()
    for side, suffix in enumerate(('BOTTOM', 'TOP')):
        assembly.SetFromNodeLabels(name='STEP5_'+suffix, nodeLabels=tuple(end_labels[side]))
    bottom_name, bottom_labels = end_labels[0][0]
    assembly.SetFromNodeLabels(name='STEP5_STOP', nodeLabels=((bottom_name, (bottom_labels[0],)),))
    stop = args.max_end_displacement_mm or length*.01
    model.StaticRiksStep(name='GMNIA', previous='Initial', nlgeom=ON,
        maxNumInc=args.max_increments, initialArcInc=args.initial_arc,
        minArcInc=args.min_arc, maxArcInc=args.max_arc, totalArcLength=1.,
        nodeOn=ON, region=assembly.sets['STEP5_STOP'], dof=3, maximumDisplacement=stop)
    for name, surface, magnitude, sign in loads:
        model.ShellEdgeLoad(name=name, createStepName='GMNIA', region=assembly.surfaces[surface],
            magnitude=magnitude, distributionType=UNIFORM, traction=GENERAL,
            directionVector=((0., 0., 0.), (0., 0., sign)), follower=OFF, resultant=OFF)
    assembly.SetFromElementLabels(name='STEP5_SHELLS', elementLabels=tuple(
        (name, tuple(e.label for e in assembly.instances[name].elements)) for name in names))
    model.FieldOutputRequest(name='ShellResponse', createStepName='GMNIA',
        region=assembly.sets['STEP5_SHELLS'], variables=('S', 'LE', 'PEEQ'),
        sectionPoints=tuple(range(1, next(iter(integration_counts))+1)), frequency=args.field_frequency)
    model.FieldOutputRequest(name='NodalResponse', createStepName='GMNIA',
        variables=('U', 'RF'), frequency=args.field_frequency)
    model.FieldOutputRequest(name='ContactResponse', createStepName='GMNIA',
        variables=('CSTRESS', 'CDISP', 'CSTATUS'), frequency=args.field_frequency)
    model.FieldOutputRequest(name='BoltResponse', createStepName='GMNIA',
        region=assembly.sets['STEP5_BOLTS'], variables=('CTF', 'CU'), frequency=args.field_frequency)
    for suffix in ('BOTTOM', 'TOP'):
        model.HistoryOutputRequest(name='End_'+suffix, createStepName='GMNIA',
            region=assembly.sets['STEP5_'+suffix], variables=('U3', 'RF3'), frequency=1)
    for name in bolts:
        model.HistoryOutputRequest(name='Force_'+name, createStepName='GMNIA',
            region=assembly.sets[name+'_CONNECTOR'],
            variables=('CTF1', 'CTF2', 'CTF3', 'CTM1', 'CTM2', 'CTM3'), frequency=1)
    model.HistoryOutputRequest(name='GlobalHistory', createStepName='GMNIA',
        variables=('ALLSE', 'ALLPD', 'ALLWK', 'ALLIE'), frequency=1)
    # Riks writes LPF automatically; LPF is not a valid CAE request identifier.
    if mesh_digest(model, names) != fingerprint:
        raise RuntimeError('Source imperfect node coordinates/connectivity changed during conversion')
    print('  Imperfect mesh preserved; preparing INP.'); sys.stdout.flush()
    info = dict(stage=5, settings=vars(args), source_step4_description=prior_description,
        reference_force_N_per_end=end_area[0]*args.reference_stress,
        end_area_mm2=end_area[0], end_area_weights=weights,
        force_formula='P=LPF*reference_force_N_per_end (no preload)',
        shortening_formula='area_weighted_mean(U3_bottom)-area_weighted_mean(U3_top)',
        stop_node=[bottom_name, bottom_labels[0]], stop_positive_U3_mm=stop,
        material='Elastic-perfectly-plastic; source E/nu preserved; no damage',
        bolts=bolts, connector='Assembled BEAM; axis 1 A->B; axis 2 projected column axis',
        source_mesh_sha256=fingerprint,
        warning='Pilot settings; no solve/convergence or physical mode classification validated. Do not remesh.')
    model.setValues(description='STEP5_GMNIA '+json.dumps(info, ensure_ascii=True))
    return info


def verify_input(path, bolts):
    with open(path) as stream:
        text = stream.read().upper()
    for pattern in (r'\*STEP[^\n]*NLGEOM=YES', r'\*STATIC[^\n]*RIKS', r'\*PLASTIC',
                    r'\*ELEMENT[^\n]*TYPE=CONN3D2', r'\*CONNECTOR SECTION', r'\bPEEQ\b', r'\bCTF1\b'):
        if not re.search(pattern, text):
            raise RuntimeError('Required INP content missing: '+pattern)
    if re.search(r'^\*(BUCKLE|MPC)\b', text, re.M):
        raise RuntimeError('Unexpected Buckle/MPC remains in STEP5 input')
    if len(re.findall(r'^\*CONNECTOR SECTION\b', text, re.M)) < 1 or bolts < 1:
        raise RuntimeError('Missing connector sections')


def build(args):
    import caeModules
    from abaqus import openMdb
    from abaqusConstants import ON, THREADS
    if not os.path.isfile(args.source_cae):
        raise ValueError('Source CAE not found: '+args.source_cae)
    if os.path.exists(args.output_dir) and (not os.path.isdir(args.output_dir) or os.listdir(args.output_dir)):
        raise ValueError('Output directory must be new or empty: '+args.output_dir)
    database = openMdb(pathName=args.source_cae)
    selected = args.models or sorted(n for n in database.models.keys() if n.startswith('STEP4_'))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select distinct STEP4_* models')
    for name in selected:
        if not name.startswith('STEP4_') or name not in database.models:
            raise ValueError('STEP4 source model not found: '+name)
    for name in list(database.jobs.keys()):
        del database.jobs[name]
    for name in list(database.models.keys()):
        if name not in selected:
            del database.models[name]
    scratch = tempfile.mkdtemp(prefix='cfs_step5_')
    previous = os.getcwd()
    outputs = []
    tag = ('%g' % args.fy).replace('.', 'p').replace('+', '')
    try:
        os.chdir(scratch)
        for i, source_name in enumerate(selected, 1):
            name = 'STEP5_'+source_name[len('STEP4_'):]+'_FY'+tag
            print('[%d/%d] BUILD %s' % (i, len(selected), name)); sys.stdout.flush()
            # changeKey can leave native-instance owner references stale in CAE.
            model = database.Model(name=name, objectToCopy=database.models[source_name])
            del database.models[source_name]
            info = prepare_model(model, args)
            job = database.Job(name=name, model=name, numCpus=args.cpus, numDomains=args.cpus,
                multiprocessingMode=THREADS, description='Build only; '+info['force_formula'])
            job.writeInput(consistencyChecking=ON)
            verify_input(name+'.inp', len(info['bolts']))
            outputs.append(name+'.inp')
            print('INP verified: %d bolts; P_ref=%g N per end' %
                  (len(info['bolts']), info['reference_force_N_per_end'])); sys.stdout.flush()
        cae_name = 'Step5_GMNIA_FY'+tag+'.cae'
        database.saveAs(pathName=os.path.join(scratch, cae_name))
        database.close()
        outputs.append(cae_name)
        os.makedirs(args.output_dir, exist_ok=True)
        for name in outputs:
            destination = os.path.join(args.output_dir, name)
            if os.path.exists(destination):
                raise ValueError('Refusing to overwrite '+destination)
            shutil.move(os.path.join(scratch, name), destination)
        print('SAVED: %s; one CAE + %d INP files. NO ANALYSIS SUBMITTED.' %
              (args.output_dir, len(selected)))
    except Exception:
        print('BUILD FAILED. Diagnostic intermediate files: '+scratch)
        raise
    else:
        # Only remove this builder's newly-created temporary workspace.
        if (os.path.commonpath([os.path.abspath(scratch), os.path.abspath(tempfile.gettempdir())])
                == os.path.abspath(tempfile.gettempdir()) and os.path.basename(scratch).startswith('cfs_step5_')):
            os.chdir(previous)
            shutil.rmtree(scratch)
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    build(parse_arguments())
