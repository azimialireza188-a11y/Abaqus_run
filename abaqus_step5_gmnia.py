# -*- coding: utf-8 -*-
"""Build only: STEP4 imperfect CAE -> STEP5 Riks CAE + one INP per selected model.

Abaqus 2024:
  abaqus cae noGUI=abaqus_step5_gmnia.py -- --source-cae "Step4_imperfections.cae"
    --output-dir "new_step5_folder" --fy 350

fy is REQUIRED (MPa). Elastic properties, thickness, mesh, imperfections, initial
BCs and general contact are inherited. Plasticity is elastic-perfectly-plastic,
with no damage, rate dependence, residual stress or bolt failure added.
All STEP4_* imperfect models plus the untouched perfect reference model are selected
unless --models explicitly lists a subset. Use the token PERFECT in --models to
select the no-imperfection reference. The compatible source is the Z-axis
four-piece model from the preceding scripts.

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

Primary post-peak stop: at the first converged increment after the peak for which
P <= postpeak_stop_ratio*Pu (default 0.70), implemented by URDFIL using the Riks
LPF and LSTOP=1. Because P=LPF*P_ref with no preload, the force ratio is exactly
the LPF ratio. A minimal NODE FILE request at every increment triggers URDFIL.
The default queue monitors ODB LPF without compiling Fortran; URDFIL is optional.

The positive-U3 monitor is disabled by default so it cannot compete with the
common 70% criterion; it is enabled only when --max-end-displacement-mm is
supplied explicitly. Max increments defaults to 5000 as a high safety cap.
A run can still fail before the 70% criterion through genuine nonconvergence or
another solver/runtime error. No equilibrium tolerances are loosened.

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
import abaqus_pipeline_contract as pipeline_contract
from abaqus_step5_progress import report

# Abaqus/CAE noGUI can execute the script without a __file__ global.
# Resolve before build changes cwd to its temporary workspace.
SCRIPT_DIR = os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename)))
STEP4_PREFIX = 'STEP4_PIPELINE_COMPATIBLE '


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
    p.add_argument('--max-increments', type=int, default=5000)
    p.add_argument('--max-end-displacement-mm', type=float)
    p.add_argument('--field-frequency', type=int, default=1)
    p.add_argument('--cpus', type=int, default=None, help='Default: all physical cores available to the solver')
    p.add_argument('--gpus', default='auto')
    p.add_argument('--stop-method', choices=['monitor', 'urdfil'], default='monitor')
    p.add_argument('--postpeak-stop-ratio', type=float, default=.70,
                   help='Stop at first converged post-peak increment with P <= ratio*Pu (default 0.70)')
    args = p.parse_args(argv)
    import runtime_resources
    try:
        policy = runtime_resources.resolve_solver_policy(runtime_resources.detect_resources(), args.cpus, args.gpus)
    except ValueError as error:
        p.error(str(error))
    args.cpus, args.gpus = policy.cpus, policy.gpus
    if args.reference_stress is None:
        args.reference_stress = args.fy
    for name in ('fy', 'reference_stress', 'initial_arc', 'min_arc', 'max_arc', 'max_end_displacement_mm'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error('--%s must be positive and finite' % name.replace('_', '-'))
    if not args.min_arc <= args.initial_arc <= args.max_arc:
        p.error('Require min-arc <= initial-arc <= max-arc')
    if args.field_frequency != 1:
        p.error('--field-frequency must be 1 for auditable LPF/field alignment and automatic plots')
    if min(args.cpus, args.max_increments, args.field_frequency) < 1:
        p.error('cpus, max-increments and field-frequency must be positive')
    if (not math.isfinite(args.postpeak_stop_ratio)
            or not 0. < args.postpeak_stop_ratio < 1.):
        p.error('--postpeak-stop-ratio must satisfy 0 < ratio < 1')
    args.source_cae = os.path.abspath(os.path.expanduser(args.source_cae))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    return args


PERFECT_TOKEN = 'PERFECT'
STEP4_MANIFEST_MARKER = 'STEP4_MANIFEST='
STOP_SUBROUTINE_NAME = 'step5_postpeak_stop.for'


def reference_step4_manifest(model):
    """Recover the exact Step-4 manifest stored on the untouched reference model."""
    text = str(model.description or '')
    index = text.rfind(STEP4_MANIFEST_MARKER)
    if index < 0:
        raise ValueError('Reference model lacks STEP4_MANIFEST provenance')
    raw = text[index+len(STEP4_MANIFEST_MARKER):].strip()
    try:
        manifest = json.loads(raw)
    except Exception as exc:
        raise ValueError('Reference STEP4_MANIFEST is not valid JSON: %s' % exc)
    payload = dict(specification=manifest, current_case={})
    pipeline_contract.validate_step4_payload(payload)
    return manifest


def perfect_step4_payload(reference_model):
    """Create Step-4-compatible zero-imperfection provenance without moving any node."""
    manifest = reference_step4_manifest(reference_model)
    case = dict(
        label='PERFECT',
        local_component_mm=0.0,
        distortional_component_mm=0.0,
        global_component_mm=0.0,
        actual_max_transverse_offset_mm=0.0,
        actual_max_normal_offset_mm=0.0,
        perfect_reference=True)
    return dict(specification=manifest, current_case=case)


def render_postpeak_urdfil(ratio):
    """Fortran URDFIL: stop at first converged post-peak LPF <= ratio*peak LPF."""
    literal = ('%.16g' % float(ratio)) + 'D0'
    return """      SUBROUTINE URDFIL(LSTOP,LOVRWRT,KSTEP,KINC,DTIME,TIME)
C
C     STEP5 GMNIA POST-PEAK STOP
C     Read LPF from increment record 2000, attribute 9 (ARRAY(11)).
C     With no preload, P/Pu = LPF/LPF_peak. Stop at the first converged
C     post-peak increment satisfying LPF <= RATIO*LPF_peak.
C
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION ARRAY(513),JRRAY(NPRECD,513),TIME(2)
      EQUIVALENCE (ARRAY(1),JRRAY(1,1))
      DOUBLE PRECISION PEAKLPF,CURRLPF,RATIO
      INTEGER PEAKINC
      LOGICAL PEAKSET,FOUND
      SAVE PEAKLPF,PEAKINC,PEAKSET
      DATA PEAKLPF /-1.D99/
      DATA PEAKINC /-1/
      DATA PEAKSET /.FALSE./
C
      RATIO=%s
      LSTOP=0
      LOVRWRT=0
      FOUND=.FALSE.
      CALL POSFIL(KSTEP,KINC,ARRAY,JRCD)
      IF (JRCD.EQ.0 .AND. JRRAY(1,2).EQ.2000) THEN
         CURRLPF=ARRAY(11)
         FOUND=.TRUE.
      END IF
      DO WHILE (.NOT.FOUND .AND. JRCD.EQ.0)
         CALL DBFILE(0,ARRAY,JRCD)
         IF (JRCD.EQ.0 .AND. JRRAY(1,2).EQ.2000) THEN
            CURRLPF=ARRAY(11)
            FOUND=.TRUE.
         END IF
      END DO
      IF (.NOT.FOUND) THEN
         WRITE(7,*) 'STEP5 STOP ERROR: LPF RECORD NOT FOUND'
         LSTOP=1
         RETURN
      END IF
C
      IF (.NOT.PEAKSET .OR. CURRLPF.GT.PEAKLPF) THEN
         PEAKLPF=CURRLPF
         PEAKINC=KINC
         PEAKSET=.TRUE.
      ELSE IF (PEAKLPF.GT.0.D0 .AND. KINC.GT.PEAKINC) THEN
         IF (CURRLPF.LE.RATIO*PEAKLPF) THEN
            LSTOP=1
            WRITE(7,9000) KINC,CURRLPF,PEAKLPF,RATIO
 9000       FORMAT(' STEP5 POSTPEAK STOP: INC=',I8,
     1             ' LPF=',1PE16.8,' PEAK=',1PE16.8,
     2             ' RATIO=',1PE12.4)
         END IF
      END IF
C
      RETURN
      END
""" % literal


def inject_urdfil_trigger(path):
    """Request minimal .fil output every increment so Abaqus calls URDFIL."""
    with open(path, 'r') as stream:
        text = stream.read()
    upper = text.upper()
    if '*NODE FILE' in upper:
        raise RuntimeError('Unexpected pre-existing NODE FILE request in STEP5 INP')
    marker = '*END STEP'
    index = upper.rfind(marker)
    if index < 0:
        raise RuntimeError('Cannot inject URDFIL trigger: *END STEP not found')
    trigger = ('** STEP5 URDFIL trigger: one-node displacement to .fil every increment\n'
               '*NODE FILE, NSET=STEP5_STOP, FREQUENCY=1\n'
               'U\n')
    text = text[:index] + trigger + text[index:]
    with open(path, 'w') as stream:
        stream.write(text)


def write_queue_batch(path, job_names, cpus='auto', subroutine_name=STOP_SUBROUTINE_NAME,
                      cae=None, script=None, ratio=.70, stop_method='monitor'):
    """One CAE runner handles monitoring, per-job plots and final comparison."""
    script = script or os.path.join(SCRIPT_DIR, 'abaqus_step5_queue_nogui.py')
    command = ('call abaqus cae noGUI="%s" -- --run-dir "%%~dp0." '
               '--cpus %s --gpus auto --ratio %g --stop-method %s --jobs %s' %
               (script, cpus, ratio, stop_method, ' '.join(job_names)))
    if cae:
        command += ' --cae "%s"' % cae
    lines = ['@echo off', 'setlocal', 'cd /d "%~dp0"',
             'set "PYTHONUNBUFFERED=1"',
             'echo STEP5 GMNIA QUEUE - POST-PEAK MONITOR AND AUTOMATIC PLOTS',
             'echo [STARTUP] Starting Abaqus/CAE queue driver; waiting for Python initialization...',
             'echo [PROGRESS] Job, stage and elapsed time will be printed; solver heartbeat every 15 seconds.',
             command, 'set "QUEUE_EXIT=%ERRORLEVEL%"',
             'echo [QUEUE EXIT] Code: %QUEUE_EXIT%',
             'if not "%QUEUE_EXIT%"=="0" echo [ATTENTION] Queue failed or a stop criterion was not reached. Check messages and STEP5_queue_status.json.',
             'endlocal & exit /b %QUEUE_EXIT%', '']
    with open(path, 'w', newline='') as stream:
        stream.write('\r\n'.join(lines))


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


def replace_bolts(model, expected_links=None):
    from abaqusConstants import BEAM, BEAM_MPC, OFF, IMPRINT, CARTESIAN
    import numpy as np
    assembly = model.rootAssembly
    if len(assembly.edges):
        raise ValueError('Unexpected existing assembly wires; use the STEP4 source')
    bolt_names = sorted(name for name in model.constraints.keys() if name.startswith('BOLT_'))
    if not bolt_names:
        raise ValueError('No BOLT_* MPC connections found')
    if expected_links is not None and len(bolt_names) != int(expected_links):
        raise ValueError('BOLT_* count differs from the reference pipeline contract')
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
                              node_b=[b[0].instanceName, b[0].label],
                              source_connection='BEAM_MPC',
                              stage5_connection='ASSEMBLED_BEAM_CONNECTOR')
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
    prior_description = model.description
    payload = pipeline_contract.parse_prefixed_json(prior_description, STEP4_PREFIX)
    specification, reference = pipeline_contract.validate_step4_payload(payload)
    names = sorted(assembly.instances.keys())
    if tuple(names) != tuple(reference['instances']):
        raise ValueError('STEP5 source instances differ from the reference pipeline P1..P4 contract')
    if any(str(e.type) != reference['element_type'] for name in names
           for e in assembly.instances[name].elements):
        raise ValueError('STEP5 source mesh differs from the reference S4R element contract')
    if set(model.steps.keys()) != {'Initial', 'Buckle'}:
        raise ValueError('Expected only Initial and Buckle inherited from STEP4')
    if 'GeneralContact' not in model.interactions or 'Hard_Frictionless' not in model.interactionProperties:
        raise ValueError('STEP5 source lost the reference general hard/frictionless contact')

    counts = specification.get('source_repository_counts') or {}
    for key in ('boundaryConditions', 'constraints', 'interactions', 'loads', 'materials', 'sections'):
        if key not in counts:
            raise ValueError('STEP4 provenance lacks source repository count: '+key)
        if len(getattr(model, key)) != int(counts[key]):
            raise ValueError('STEP5 source repository differs from STEP4/reference for '+key)
    if len(model.boundaryConditions) != int(reference['boundary_conditions']):
        raise ValueError('STEP5 source boundary conditions differ from the reference pipeline')
    if len(model.loads) != int(reference['loads']):
        raise ValueError('STEP5 source loads differ from the reference pipeline')
    if len(model.constraints) != int(reference['rigid_links']):
        raise ValueError('STEP5 source BEAM_MPC count differs from the reference pipeline')
    if reference['connection_model'] != 'BEAM_MPC':
        raise ValueError('STEP5 requires the production BEAM_MPC reference source')

    fingerprint = mesh_digest(model, names)
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
    bolts = replace_bolts(model, expected_links=reference['expected_links'])
    print('  Defining Riks loading and response outputs...'); sys.stdout.flush()
    for side, suffix in enumerate(('BOTTOM', 'TOP')):
        assembly.SetFromNodeLabels(name='STEP5_'+suffix, nodeLabels=tuple(end_labels[side]))
    bottom_name, bottom_labels = end_labels[0][0]
    assembly.SetFromNodeLabels(name='STEP5_STOP', nodeLabels=((bottom_name, (bottom_labels[0],)),))
    stop = args.max_end_displacement_mm
    if stop is None:
        model.StaticRiksStep(name='GMNIA', previous='Initial', nlgeom=ON,
            maxNumInc=args.max_increments, initialArcInc=args.initial_arc,
            minArcInc=args.min_arc, maxArcInc=args.max_arc, totalArcLength=1.,
            nodeOn=OFF)
    else:
        model.StaticRiksStep(name='GMNIA', previous='Initial', nlgeom=ON,
            maxNumInc=args.max_increments, initialArcInc=args.initial_arc,
            minArcInc=args.min_arc, maxArcInc=args.max_arc, totalArcLength=1.,
            nodeOn=ON, region=assembly.sets['STEP5_STOP'], dof=3,
            maximumDisplacement=stop)
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
    info = dict(stage=5, settings=vars(args),
        source_step4=dict(case=payload['current_case'], specification=specification),
        reference_pipeline=reference,
        reference_force_N_per_end=end_area[0]*args.reference_stress,
        end_area_mm2=end_area[0], end_area_weights=weights,
        force_formula='P=LPF*reference_force_N_per_end (no preload)',
        shortening_formula='area_weighted_mean(U3_bottom)-area_weighted_mean(U3_top)',
        postpeak_stop=dict(
            ratio=args.postpeak_stop_ratio,
            criterion='first converged post-peak increment with P <= ratio*Pu',
            equivalent_lpf_criterion='LPF <= ratio*max_previous_LPF because P=LPF*P_ref and P0=0',
            implementation=('ODB LPF polling and native Abaqus terminate; first observed crossing'
                            if args.stop_method == 'monitor' else 'URDFIL LSTOP=1 triggered by NODE FILE frequency=1'),
            user_subroutine=STOP_SUBROUTINE_NAME),
        stop_node=[bottom_name, bottom_labels[0]], stop_positive_U3_mm=stop,
        material='Elastic-perfectly-plastic; source E/nu preserved; no damage',
        bolts=bolts,
        connection_transition=dict(
            source='BEAM_MPC from abaqus_complete_model_m20.py',
            stage5='Assembled BEAM connector on the exact same endpoint node pairs',
            reason='retain rigid beam kinematic intent while enabling connector force/moment output',
            added_compliance='none intentionally introduced'),
        source_mesh_sha256=fingerprint,
        compatibility_checks=dict(
            instances=names, element_type=reference['element_type'],
            source_rigid_links=int(reference['rigid_links']),
            source_boundary_conditions=int(reference['boundary_conditions']),
            source_loads=int(reference['loads']),
            source_contact=reference['shell_contact']),
        warning='Pilot Riks settings; no solve/convergence or physical mode classification validated. Do not remesh.')
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
    if not re.search(r'^\*NODE FILE[^\n]*NSET=STEP5_STOP[^\n]*FREQUENCY=1', text, re.M):
        raise RuntimeError('URDFIL trigger NODE FILE request is missing from STEP5 input')
    if len(re.findall(r'^\*CONNECTOR SECTION\b', text, re.M)) < 1 or bolts < 1:
        raise RuntimeError('Missing connector sections')


def build(args):
    report('BUILD START', 'Loading Abaqus modules')
    import caeModules
    from abaqus import openMdb
    from abaqusConstants import ON, PERCENTAGE, FULL, SINGLE
    if not os.path.isfile(args.source_cae):
        raise ValueError('Source CAE not found: '+args.source_cae)
    if os.path.exists(args.output_dir) and (not os.path.isdir(args.output_dir) or os.listdir(args.output_dir)):
        raise ValueError('Output directory must be new or empty: '+args.output_dir)
    report('OPEN CAE', args.source_cae)
    database = openMdb(pathName=args.source_cae)
    report('VALIDATE', 'Checking Step4 models and reference provenance')
    imperfect = sorted(n for n in database.models.keys() if n.startswith('STEP4_'))
    reference_candidates = [n for n in database.models.keys()
        if not n.startswith('STEP4_') and STEP4_MANIFEST_MARKER in str(database.models[n].description or '')]
    if len(reference_candidates) != 1:
        raise ValueError('Expected exactly one untouched Step-4 reference model; found %d' %
                         len(reference_candidates))
    perfect_source = reference_candidates[0]
    # Validate the untouched reference provenance before using it as PERFECT.
    reference_step4_manifest(database.models[perfect_source])

    selected = list(args.models) if args.models else [PERFECT_TOKEN] + imperfect
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select distinct STEP4_* models and/or the PERFECT token')
    for name in selected:
        if name == PERFECT_TOKEN:
            continue
        if not name.startswith('STEP4_') or name not in database.models:
            raise ValueError('STEP4 source model not found: '+name)
    for name in list(database.jobs.keys()):
        del database.jobs[name]
    keep = set(n for n in selected if n != PERFECT_TOKEN)
    if PERFECT_TOKEN in selected:
        keep.add(perfect_source)
    for name in list(database.models.keys()):
        if name not in keep:
            del database.models[name]
    scratch = tempfile.mkdtemp(prefix='cfs_step5_')
    previous = os.getcwd()
    outputs = []
    job_names = []
    tag = ('%g' % args.fy).replace('.', 'p').replace('+', '')
    final_user_subroutine = os.path.join(args.output_dir, STOP_SUBROUTINE_NAME)
    try:
        os.chdir(scratch)
        with open(STOP_SUBROUTINE_NAME, 'w') as stream:
            stream.write(render_postpeak_urdfil(args.postpeak_stop_ratio))
        outputs.append(STOP_SUBROUTINE_NAME)
        with open('abaqus_v6.env','w') as stream:
            stream.write('standard_parallel = ALL\ngpus = %d\ncpus = %d\nmemory = \"100%%\"\n' % (args.gpus,args.cpus))
        outputs.append('abaqus_v6.env')

        for i, source_name in enumerate(selected, 1):
            report('BUILD %d/%d' % (i, len(selected)), 'Copying source model '+source_name)
            if source_name == PERFECT_TOKEN:
                name = 'STEP5_PERFECT_FY'+tag
                source_model = database.models[perfect_source]
                model = database.Model(name=name, objectToCopy=source_model)
                model.setValues(description=STEP4_PREFIX+json.dumps(
                    perfect_step4_payload(source_model), ensure_ascii=True))
            else:
                name = 'STEP5_'+source_name[len('STEP4_'):]+'_FY'+tag
                model = database.Model(name=name, objectToCopy=database.models[source_name])
            report('PREPARE', name+' | materials, connectors, Riks step and outputs')
            info = prepare_model(model, args)
            nodal_precision = str(info['reference_pipeline'].get('nodal_precision', 'full')).lower()
            if nodal_precision not in ('full', 'single'):
                raise ValueError('Unsupported inherited nodal precision: '+nodal_precision)
            job = database.Job(
                name=name, model=name, numCpus=args.cpus, numDomains=args.cpus,
                memory=100, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=False,
                nodalOutputPrecision=FULL if nodal_precision == 'full' else SINGLE,
                userSubroutine=final_user_subroutine if args.stop_method == 'urdfil' else '',
                description='Build only; reference-compatible STEP5; '+info['force_formula'])
            report('WRITE INP', name)
            job.writeInput(consistencyChecking=ON)
            report('VERIFY INP', name)
            inject_urdfil_trigger(name+'.inp')
            verify_input(name+'.inp', len(info['bolts']))
            outputs.append(name+'.inp')
            job_names.append(name)
            print('INP verified: %d bolts; P_ref=%g N per end; postpeak stop=%g%% Pu' %
                  (len(info['bolts']), info['reference_force_N_per_end'],
                   100.*args.postpeak_stop_ratio)); sys.stdout.flush()

        # Keep only final STEP5 models in the delivered CAE.
        for source_name in list(database.models.keys()):
            if not source_name.startswith('STEP5_'):
                del database.models[source_name]

        ratio_tag = ('%g' % (100.*args.postpeak_stop_ratio)).replace('.', 'p')
        queue_name = 'run_step5_queue_'+ratio_tag+'_pct.bat'
        report('WRITE BAT', queue_name)
        write_queue_batch(queue_name, job_names, 'auto', ratio=args.postpeak_stop_ratio, stop_method=args.stop_method)
        outputs.append(queue_name)

        cae_name = 'Step5_GMNIA_FY'+tag+'.cae'
        report('SAVE CAE', cae_name)
        database.saveAs(pathName=os.path.join(scratch, cae_name))
        database.close()
        outputs.append(cae_name)
        os.makedirs(args.output_dir, exist_ok=True)
        report('COPY OUTPUTS', args.output_dir)
        for name in outputs:
            destination = os.path.join(args.output_dir, name)
            if os.path.exists(destination):
                raise ValueError('Refusing to overwrite '+destination)
            shutil.move(os.path.join(scratch, name), destination)
        print('SAVED: %s; one CAE + %d INP files + URDFIL + sequential queue. NO ANALYSIS SUBMITTED.' %
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
