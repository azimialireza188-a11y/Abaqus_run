# -*- coding: utf-8 -*-
"""STEP 4 ONLY: create a new CAE containing documented imperfect meshes from the exact completed reference pipeline run.

Reference identity comes from run-dir/pipeline_status.json; arbitrary CAE/ODB glob selection is forbidden.\n\nSuggestions (read existing step-3 reports; no CAE/solver needed):
  abaqus python abaqus_step4_imperfections.py --run-dir "completed run" --suggest

Build after YOU confirm mode IDs and amplitudes:
  abaqus cae noGUI=abaqus_step4_imperfections.py -- --run-dir "completed run"
    --local-mode INTEGER --dist-mode INTEGER --local-high-t FACTOR
    --dist-mm MILLIMETRES --output-cae "new output.cae"

local-high-t and dist-mm have NO assumed defaults. The low local level defaults
to 0.34*t (the research plan). With --local-metric normal, its scale is the max
absolute displacement along averaged original shell normals on the gauge nodes.
The default gauge is all shell nodes; use --local-gauge INSTANCE:LABEL,... to
select the stiffened-panel measurement points relevant to the cited statistic.
Numerical modal scaling alone does not establish statistical equivalence to a
published imperfection measurement. --dist-gauge uses max transverse norm.

Seven default cases: L_low, L_high, D, LD_pp, LD_pm, G1000, G3000.
Optional LD_mp and LD_mm complete all four local/distortional sign combinations.
Use --cases all for all nine, or --cases L_low,D,LD_pm,G1000 for a documented subset.
Each component is normalized independently, THEN combined without rescaling.
Global bow = sin(pi*(z-z0)/L) in the --global-angle-deg direction (default global
X for a Z-axis column), with analytical peaks L/1000 and L/3000. This is not an
automatically chosen weak axis or an imported global eigenmode.

Only transverse eigenvector translations modify coordinates; axial U and nodal
rotations are not used as geometric imperfections. Ends stay at original nodes.
Native part geometry remains perfect; DISPLAY THE MESH to inspect imperfections.
Do NOT remesh/regenerate geometry: doing so can erase the imposed nodal offsets.
The source model's sections, contact, BCs, MPCs and steps/loads are preserved in
each copy. Existing Buckle steps are reference setup, NOT a GMNIA/Riks analysis.
No material plasticity, new step, INP, ODB or solver job is created/submitted.
All saved Job objects are removed from the NEW database to avoid submitting a
perfect reference job by mistake. The original CAE/ODB files are never saved over.
The full input specification and each case's amplitudes are stored in the Model
description inside the new CAE. No separate output data files are required.

Scope: four matching, straight, meshed shell instances, axis aligned with X/Y/Z;
uniform thickness. A matching step-3 CAE and ODB are required. Abaqus 2024/Python 3.
"""
import argparse
import builtins as python_builtins
import csv
import json
import math
import os
import re
import sys
import numpy as np
import abaqus_pipeline_contract as pipeline_contract

DEFAULT_CASES = ('L_low', 'L_high', 'D', 'LD_pp', 'LD_pm', 'G1000', 'G3000')


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
                clean.append(argv[i])
                i += 1
        argv = clean
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--suggest', action='store_true', help='Report candidates only; never select a mode automatically')
    p.add_argument('--source-cae', help='Optional explicit reference CAE path; must equal the artifact resolved from --run-dir')
    p.add_argument('--odb', help='Optional explicit reference ODB path; must equal the artifact resolved from --run-dir')
    p.add_argument('--model', help='Optional explicit model name; must equal pipeline_status build.job_name')
    p.add_argument('--step', default='Buckle')
    p.add_argument('--instances', nargs=4, help='Optional explicit instances; must be exactly P1 P2 P3 P4')
    p.add_argument('--local-mode', type=int)
    p.add_argument('--dist-mode', type=int)
    p.add_argument('--local-low-t', type=float, default=.34)
    p.add_argument('--local-high-t', type=float)
    p.add_argument('--dist-mm', type=float)
    p.add_argument('--local-metric', choices=('normal', 'transverse'), default='normal')
    p.add_argument('--local-gauge', help='Comma-separated INSTANCE:LABEL pairs; otherwise all selected nodes')
    p.add_argument('--dist-gauge', help='Comma-separated INSTANCE:LABEL pairs; otherwise all selected nodes')
    p.add_argument('--axis', choices=('x', 'y', 'z'), default='z')
    p.add_argument('--global-angle-deg', type=float, default=0.)
    p.add_argument('--cases', default=','.join(DEFAULT_CASES))
    p.add_argument('--output-cae')
    args = p.parse_args(argv)
    args.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if not args.suggest:
        for name in ('local_mode', 'dist_mode', 'local_high_t', 'dist_mm', 'output_cae'):
            if getattr(args, name) is None:
                p.error('--%s must be supplied explicitly; use --suggest first if needed' % name.replace('_', '-'))
        if args.local_mode < 1 or args.dist_mode < 1 or args.local_mode == args.dist_mode:
            p.error('Choose two distinct positive mode numbers confirmed from step 3')
        for name in ('local_low_t', 'local_high_t', 'dist_mm'):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                p.error('--%s must be positive and finite' % name.replace('_', '-'))
        if args.local_high_t <= args.local_low_t:
            p.error('--local-high-t must exceed --local-low-t')
        if not math.isfinite(args.global_angle_deg):
            p.error('--global-angle-deg must be finite')
        args.output_cae = os.path.abspath(os.path.expanduser(args.output_cae))
        if not args.output_cae.lower().endswith('.cae'):
            args.output_cae += '.cae'
    return args


def suggest_modes(run_dir):
    reference = pipeline_contract.load_reference_run(run_dir, require_completed=True)
    path = os.path.join(reference['run_dir'],
                        reference['job_name']+'_modal_wavelengths_enhanced_modes.csv')
    if not os.path.isfile(path):
        print('NO AUTOMATIC CANDIDATES: the reference enhanced-mode CSV is missing: '+path)
        print('Run the normal reference postprocessing first, or inspect the matching ODB manually. No CAE was created.')
        return
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    print('Reference job: '+reference['job_name'])
    print('Source: '+path)
    print('Heuristic suggestions only, NOT confirmed physical mode families. Verify full shapes in the step-3 ODB.')
    for family, flag in (('Local-like', '--local-mode'), ('Distortional-like', '--dist-mode')):
        candidates = [r for r in rows if r.get('family') == family and float(r['eigenvalue']) > 0]
        candidates.sort(key=lambda r: float(r['eigenvalue']))
        if not candidates:
            print('NO %s CANDIDATE in the step-3 classification; this does NOT prove that the ODB contains no such mode.' % family.upper())
            print('Manual mode-shape inspection is required before supplying '+flag)
        else:
            print('%s candidates (lowest positive eigenvalues first):' % family)
            for row in candidates[:5]:
                print('  mode=%s, eigenvalue=%s, half-wave=%s mm, flags=%s' %
                    (row['mode'], row['eigenvalue'], row['half_wavelength_mm'], row.get('enhanced_flags', row.get('status', ''))))
    print('No mode/amplitude was selected. Supply --local-high-t and --dist-mm explicitly when building.')


def case_definitions(thickness, low_factor, high_factor, dist_mm, length):
    low, high = thickness*low_factor, thickness*high_factor
    return dict(L_low=(low, 0., 0.), L_high=(high, 0., 0.), D=(0., dist_mm, 0.),
        LD_pp=(low, dist_mm, 0.), LD_pm=(low, -dist_mm, 0.),
        LD_mp=(-low, dist_mm, 0.), LD_mm=(-low, -dist_mm, 0.),
        G1000=(0., 0., length/1000.), G3000=(0., 0., length/3000.))


def normalize_mode(displacement, normals, axis, gauge, metric):
    u = np.asarray(displacement, dtype=float).copy()
    if not np.all(np.isfinite(u)):
        raise ValueError('Non-finite mode displacements')
    u[:, axis] = 0.
    gauge = np.asarray(gauge, dtype=int)
    if not len(gauge):
        raise ValueError('Empty amplitude gauge')
    if metric == 'normal':
        signed = np.sum(u*normals, axis=1)[gauge]
        peak = int(np.argmax(np.abs(signed)))
        denominator = abs(float(signed[peak]))
        sign = 1. if signed[peak] >= 0 else -1.
    else:
        denominator = float(np.max(np.linalg.norm(u[gauge], axis=1)))
        signed = u[gauge].ravel()
        peak = int(np.argmax(np.abs(signed)))
        sign = 1. if signed[peak] >= 0 else -1.
    if denominator <= np.finfo(float).tiny:
        raise ValueError('Selected mode has no displacement in the chosen gauge/metric')
    return u*sign/denominator, dict(metric=metric, raw_denominator=denominator,
        eigenvector_sign_multiplier=sign, gauge_node_count=len(gauge))


def global_bow(coordinates, axis, angle_degrees):
    coordinates = np.asarray(coordinates)
    z = coordinates[:, axis]
    length = float(z.max()-z.min())
    if length <= 0:
        raise ValueError('Invalid column length')
    transverse = [i for i in range(3) if i != axis]
    wave = np.sin(np.pi*(z-z.min())/length)
    wave[(z == z.min()) | (z == z.max())] = 0.
    result = np.zeros_like(coordinates, dtype=float)
    angle = math.radians(angle_degrees)
    result[:, transverse[0]] = wave*math.cos(angle)
    result[:, transverse[1]] = wave*math.sin(angle)
    return result


def shell_instances(model):
    return sorted(name for name, inst in model.rootAssembly.instances.items()
                  if len(inst.elements) and all(str(e.type).startswith(('S3', 'S4', 'S6', 'S8', 'S9', 'STRI')) for e in inst.elements))


def gauge_indices(text, keys):
    if not text:
        return list(range(len(keys)))
    lookup = {(name.upper(), label): i for i, (name, label) in enumerate(keys)}
    result = []
    for token in text.split(','):
        name, label = token.strip().rsplit(':', 1)
        key = (name.upper(), int(label))
        if key not in lookup:
            raise ValueError('Amplitude gauge node not found: '+token)
        result.append(lookup[key])
    return sorted(set(result))


def read_modes_and_mesh(model, odb, names, args):
    axis = 'xyz'.index(args.axis)
    keys, xyz, normals, element_topology = [], [], [], {}
    cae_instances = model.rootAssembly.instances
    odb_names = {name.upper(): name for name in odb.rootAssembly.instances.keys()}
    instance_map = {}
    for name in names:
        if name.upper() not in odb_names:
            raise ValueError('Instance missing from ODB: '+name)
        odb_name = odb_names[name.upper()]
        instance_map[odb_name] = name
        ci, oi = cae_instances[name], odb.rootAssembly.instances[odb_name]
        cn = {node.label: np.asarray(node.coordinates, dtype=float) for node in ci.nodes}
        on = {node.label: np.asarray(node.coordinates, dtype=float) for node in oi.nodes}
        if set(cn) != set(on):
            raise ValueError('CAE/ODB node labels differ in '+name)
        tolerance = max(1e-5, 1e-6*max(np.max(np.abs(v)) for v in cn.values()))
        if any(np.max(np.abs(cn[label]-on[label])) > tolerance for label in cn):
            raise ValueError('CAE/ODB coordinates differ; use the matching PERFECT step-3 model: '+name)
        ce = {e.label: tuple(n.label for n in e.getNodes()) for e in ci.elements}
        oe = {e.label: tuple(e.connectivity) for e in oi.elements}
        if ce != oe:
            raise ValueError('CAE/ODB element connectivity differs in '+name)
        element_topology[name] = ce
        sums = {label: np.zeros(3) for label in cn}
        for element in oi.elements:
            kind = str(element.type)
            corners = element.connectivity[:3 if kind.startswith(('S3', 'S6', 'STRI')) else 4]
            points = [on[label] for label in corners]
            # CAE noGUI injects its own sum into the script's global namespace.
            vector = python_builtins.sum((np.cross(points[i]-points[0], points[i+1]-points[0])
                          for i in range(1, len(points)-1)), np.zeros(3))
            for label in element.connectivity:
                sums[label] += vector
        for label in sorted(cn):
            keys.append((name, label))
            xyz.append(cn[label])
            norm = np.linalg.norm(sums[label])
            normals.append(sums[label]/norm if norm else np.zeros(3))
    xyz, normals = np.asarray(xyz), np.asarray(normals)
    lookup = {key: i for i, key in enumerate(keys)}
    frames = {}
    if args.step not in odb.steps:
        raise ValueError('Buckling step not found: '+args.step)
    for frame in odb.steps[args.step].frames:
        mode = int(getattr(frame, 'mode', 0) or 0)
        if mode <= 0:
            match = re.search(r'\bMode\s*[:=]?\s*(\d+)', str(frame.description), re.I)
            mode = int(match.group(1)) if match else 0
        if mode > 0:
            if mode in frames:
                raise ValueError('Duplicate mode IDs in selected step')
            frames[mode] = frame
    fields, descriptions = {}, {}
    for mode in (args.local_mode, args.dist_mode):
        if mode not in frames:
            raise ValueError('Requested mode %d absent from step-3 ODB' % mode)
        frame = frames[mode]
        if 'U' not in frame.fieldOutputs:
            raise ValueError('Mode has no nodal U field: %d' % mode)
        values = np.full((len(keys), 3), np.nan)
        for value in frame.fieldOutputs['U'].values:
            if value.instance is None or value.instance.name not in instance_map:
                continue
            key = (instance_map[value.instance.name], value.nodeLabel)
            if key not in lookup:
                continue
            double = str(value.precision) == 'DOUBLE_PRECISION'
            system = value.localCoordSystemDouble if double else value.localCoordSystem
            if system is not None and len(system):
                raise ValueError('U uses a local coordinate system; export global nodal U first')
            values[lookup[key]] = value.dataDouble if double else value.data
        if not np.all(np.isfinite(values)):
            raise ValueError('Incomplete U data for mode %d' % mode)
        values[:, axis] = 0.
        z = xyz[:, axis]
        tolerance = max(1e-5, 1e-6*float(z.max()-z.min()))
        ends = (abs(z-z.min()) <= tolerance) | (abs(z-z.max()) <= tolerance)
        peak = np.max(np.linalg.norm(values, axis=1))
        if np.max(np.linalg.norm(values[ends], axis=1)) > 1e-4*peak:
            raise ValueError('Mode %d does not satisfy zero transverse end displacement' % mode)
        values[ends] = 0.
        fields[mode] = values
        descriptions[str(mode)] = frame.description
    return keys, xyz, normals, fields, descriptions, element_topology


def apply_offsets(model, names, keys, reference_coordinates, offsets, topology):
    from abaqusConstants import ON, OFF
    import meshEdit
    assembly = model.rootAssembly
    dependent = tuple(assembly.instances[name] for name in names if assembly.instances[name].dependent == ON)
    if dependent:
        assembly.makeIndependent(instances=dependent)
    index = {key: i for i, key in enumerate(keys)}
    for name in names:
        instance = assembly.instances[name]
        nodes = instance.nodes  # MeshNodeArray is ordered by mesh index, not node label.
        rows = [index[(name, node.label)] for node in nodes]
        coordinates = reference_coordinates[rows]+offsets[rows]
        assembly.editNode(nodes=nodes, coordinates=tuple(tuple(float(v) for v in row) for row in coordinates),
                          projectToGeometry=OFF)
        actual = np.asarray([node.coordinates for node in assembly.instances[name].nodes])
        tolerance = max(1e-6, 1e-7*float(np.max(np.abs(reference_coordinates))))
        if actual.shape != coordinates.shape or np.max(np.abs(actual-coordinates)) > tolerance:
            raise RuntimeError('Edited node coordinates did not match the requested imperfection: '+name)
        current = {e.label: tuple(n.label for n in e.getNodes()) for e in assembly.instances[name].elements}
        if current != topology[name]:
            raise RuntimeError('Mesh connectivity changed while applying offsets: '+name)


def build(args):
    import caeModules
    from abaqus import openMdb, session
    from abaqusConstants import ON, BEAM_MPC
    from odbAccess import openOdb

    reference = pipeline_contract.load_reference_run(args.run_dir, require_completed=True)
    source_cae = pipeline_contract.resolve_reference_artifact(reference, args.source_cae, 'cae')
    odb_path = pipeline_contract.resolve_reference_artifact(reference, args.odb, 'odb')
    if os.path.exists(args.output_cae):
        raise ValueError('Output CAE already exists; select a new filename: '+args.output_cae)

    database = openMdb(pathName=source_cae)
    model_name = args.model or reference['job_name']
    if model_name != reference['job_name']:
        raise ValueError('STEP4 must use the reference pipeline model %s, not %s' %
                         (reference['job_name'], model_name))
    if model_name not in database.models:
        raise ValueError('Reference model is missing from source CAE: '+model_name)
    source = database.models[model_name]
    names = list(args.instances or reference['instances'])
    if tuple(names) != tuple(reference['instances']):
        raise ValueError('STEP4 instances must be exactly '+','.join(reference['instances']))
    if shell_instances(source) != sorted(reference['instances']):
        raise ValueError('Source CAE shell instances differ from the reference pipeline')

    # Enforce the exact upstream production contract before perturbing any node.
    if len(source.boundaryConditions) != int(reference['build']['boundary_conditions']):
        raise ValueError('Source CAE boundary conditions differ from pipeline_status.json')
    if len(source.loads) != int(reference['build']['loads']):
        raise ValueError('Source CAE loads differ from pipeline_status.json')
    if len(source.constraints) != int(reference['build']['rigid_links']):
        raise ValueError('Source CAE bolt constraint count differs from pipeline_status.json')
    bolt_names = sorted(name for name in source.constraints if name.startswith('BOLT_'))
    if len(bolt_names) != int(reference['source_inputs']['expected_links']):
        raise ValueError('Source CAE BOLT_* count differs from the exported seam/bolt definition')
    if any(getattr(source.constraints[name], 'mpcType', None) != BEAM_MPC for name in bolt_names):
        raise ValueError('Source CAE no longer uses the reference BEAM_MPC bolt model')
    if 'GeneralContact' not in source.interactions or 'Hard_Frictionless' not in source.interactionProperties:
        raise ValueError('Source CAE no longer contains the reference general hard/frictionless contact')
    thicknesses = set()
    for name in names:
        instance = source.rootAssembly.instances[name]
        assignments = source.parts[instance.partName].sectionAssignments
        if not assignments:
            raise ValueError('Missing shell section assignment: '+name)
        for assignment in assignments:
            thickness = getattr(source.sections[assignment.sectionName], 'thickness', None)
            if thickness is None or thickness <= 0:
                raise ValueError('A constant positive shell thickness is required')
            thicknesses.add(float(thickness))
    if len(thicknesses) != 1:
        raise ValueError('This step-4 builder requires uniform shell thickness')
    thickness = next(iter(thicknesses))
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        keys, xyz, normals, raw, descriptions, topology = read_modes_and_mesh(source, odb, names, args)
    finally:
        odb.close()
    axis = 'xyz'.index(args.axis)
    length = float(np.ptp(xyz[:, axis]))
    for name in names:
        z = xyz[[i for i, key in enumerate(keys) if key[0] == name], axis]
        if not np.allclose([z.min(), z.max()], [xyz[:, axis].min(), xyz[:, axis].max()], atol=1e-5, rtol=0):
            raise ValueError('The four pieces must have common ends')
    local, local_meta = normalize_mode(raw[args.local_mode], normals, axis,
        gauge_indices(args.local_gauge, keys), args.local_metric)
    dist, dist_meta = normalize_mode(raw[args.dist_mode], normals, axis,
        gauge_indices(args.dist_gauge, keys), 'transverse')
    global_shape = global_bow(xyz, axis, args.global_angle_deg)
    cases = case_definitions(thickness, args.local_low_t, args.local_high_t, args.dist_mm, length)
    selected = list(cases) if args.cases == 'all' else [name.strip() for name in args.cases.split(',')]
    if len(set(selected)) != len(selected) or not selected or any(name not in cases for name in selected):
        raise ValueError('Invalid/duplicate --cases. Available: '+','.join(cases))
    case_names = ['STEP4_'+name for name in selected]
    if any(name in database.models for name in case_names):
        raise ValueError('STEP4 model names already exist in source; start from the perfect step-3 CAE')
    source_counts = {key: len(getattr(source, key)) for key in
        ('boundaryConditions', 'constraints', 'interactions', 'loads', 'materials', 'sections')}
    manifest = dict(stage=4, source_cae=source_cae, source_model=model_name, source_odb=odb_path,
        source_odb_size=os.path.getsize(odb_path), source_odb_mtime=os.path.getmtime(odb_path),
        reference_pipeline=pipeline_contract.reference_snapshot(reference, include_hashes=True),
        source_repository_counts=source_counts,
        settings=vars(args), thickness_mm=thickness, length_mm=length, source_frames=descriptions,
        local_normalization=local_meta, distortional_normalization=dist_meta,
        global_shape='Analytical half-sine bow; prescribed transverse direction, not inferred weak axis',
        convention='Only transverse nodal translations; independent component normalization; no rescaling after addition',
        scope='Imperfection specimens only. Existing elastic steps/loads retained as reference; no GMNIA setup or solver jobs.',
        measurement_warning='0.34t is the plan input, not proof of statistical equivalence. Select a relevant local gauge.',
        mesh_warning='Imperfections are in node coordinates. Show mesh. Do not remesh.', cases={})
    counts = dict(source_counts)
    for case_name in selected:
        a_local, a_dist, a_global = cases[case_name]
        offsets = a_local*local+a_dist*dist+a_global*global_shape
        name = 'STEP4_'+case_name
        print('Creating %s: local=%g mm, distortional=%g mm, global=%g mm' % (name, a_local, a_dist, a_global))
        sys.stdout.flush()
        model = database.Model(name=name, objectToCopy=source)
        apply_offsets(model, names, keys, xyz, offsets, topology)
        if any(len(getattr(model, key)) != value for key, value in counts.items()):
            raise RuntimeError('Copied model lost a boundary condition, connection or property')
        item = dict(local_component_mm=a_local, distortional_component_mm=a_dist, global_component_mm=a_global,
            actual_max_transverse_offset_mm=float(np.max(np.linalg.norm(offsets, axis=1))),
            actual_max_normal_offset_mm=float(np.max(np.abs(np.sum(offsets*normals, axis=1)))))
        manifest['cases'][name] = item
    for name in case_names:
        database.models[name].setValues(description='STEP4_PIPELINE_COMPATIBLE '+json.dumps(
            dict(specification=manifest, current_case=manifest['cases'][name]), ensure_ascii=True))
    # Remove jobs only in the new in-memory database; source disk file is unchanged.
    for name in list(database.jobs.keys()):
        del database.jobs[name]
    source.setValues(description=str(source.description or '')+'\nSTEP4_MANIFEST='+json.dumps(manifest, ensure_ascii=True))
    os.makedirs(os.path.dirname(args.output_cae), exist_ok=True)
    if session.viewports:
        viewport = session.viewports[session.currentViewportName]
        viewport.setValues(displayedObject=database.models[case_names[0]].rootAssembly)
        viewport.assemblyDisplay.setValues(mesh=ON)
        viewport.view.fitView()
    database.saveAs(pathName=args.output_cae)
    print('SAVED: %s (%d imperfect models plus reference). No analysis submitted.' % (args.output_cae, len(selected)))
    return manifest


def main(argv=None):
    args = parse_arguments(argv)
    if args.suggest:
        return suggest_modes(args.run_dir)
    return build(args)


if __name__ == '__main__':
    main()
