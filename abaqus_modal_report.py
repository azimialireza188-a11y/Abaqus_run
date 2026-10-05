# -*- coding: utf-8 -*-
"""Enhanced modal reports AFTER abaqus_modal_wavelengths.py; never submit a job.

abaqus python abaqus_modal_report.py "RUN/model_modal_wavelengths_report.json"

Classification is a geometric displacement proxy, NOT cFSM/GBT identification.
All wavelength and eigenvalue data come from the existing CSV/report; the ODB
is read once for cross-sectional deformation diagnostics. Requires Abaqus Python
3, NumPy and Matplotlib. Output: 3 PNGs, interactive offline HTML, CSVs and JSON.
"""
import argparse
import base64
import csv
import html
import importlib.util
import json
import math
import os
import sys
from collections import Counter
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename)))
_walls_spec = importlib.util.spec_from_file_location(
    'pipeline_physical_walls', os.path.join(SCRIPT_DIR, 'abaqus_physical_walls.py'))
_walls_module = importlib.util.module_from_spec(_walls_spec)
_walls_spec.loader.exec_module(_walls_module)
physical_wall_layout = _walls_module.physical_wall_layout
wall_kinematics = _walls_module.wall_kinematics
COLORS = {'Global-like': '#2369b0', 'Distortional-like': '#e58a16',
          'Local-like': '#29946b', 'Assembly-like': '#b060c8', 'Other-like': '#9b6b43',
          'Mixed': '#8b58a0', 'Unresolved': '#777777'}
LIMITATION = ('Geometric displacement classification only; not cFSM/GBT or strain-energy '
    'participation. Whole-section rigid motion is fitted from fold-line/anchor motion, not '
    'from panel interiors. Independent rigid motion of built-up pieces is reported as an '
    'Assembly-like component and is excluded from the L/D/G denominator. Anchor motion that '
    'changes panel chord length is reported as Other-like transverse extension and is also '
    'excluded. Local plate bending is measured first from normal displacement relative '
    'to endpoint-driven wall motion that reproduces rigid rotation of initially curved walls. '
    'Compact source bends delimit walls; smooth wall curvature remains inside each wall. '
    'Distortional proxy is then driven only by the remaining '
    'inextensional fold/coarse motion. The wall geometry comes from '
    'builtup_segments.csv when available. Older runs without persisted wall geometry use a mesh fallback and are '
    'flagged accordingly. Inspect the displayed shapes.')


def orth(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    matrix = matrix[:, norms > 1e-12]
    if not matrix.shape[1]:
        return np.empty((matrix.shape[0], 0))
    matrix = matrix / np.linalg.norm(matrix, axis=0)
    u, singular, unused = np.linalg.svd(matrix, full_matrices=False)
    return u[:, singular > singular[0]*1e-10]


class SectionProjector:
    """Curvature-aware geometric screening split for built-up open sections.

    The key rule is that Local is measured first from bending of a physical
    wall relative to endpoint-driven motion of its curved reference. Therefore a
    local plate buckle cannot be consumed by a coarse Distortional
    interpolation merely because the corner/fold nodes also move.

      L: normal within-wall bending after rigid-exact endpoint interpolation,
      G: whole-section rigid motion of physical fold lines,
      A: independent rigid motion of built-up pieces after G,
      O: first-order wall-chord extension/shear-like fold motion,
      D: remaining inextensional fold-line/coarse-wall motion.

    physical_segments is optional. When supplied it is the preferred source of
    wall locations and comes from builtup_segments.csv through the one-command
    pipeline. Older runs fall back to a conservative mesh straight-run detector.
    """

    def __init__(self, xy, edges, pieces, weights, corner_angle=15.,
                 physical_segments=None, wall_angle_deg=3.0, bend_radius_fraction=.04,
                 compute_bases=True):
        self.xy = np.asarray(xy, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.pieces = np.asarray(pieces)
        count = len(self.xy)
        if self.weights.shape != (count,) or np.any(self.weights <= 0):
            raise ValueError('Positive section weights required')
        self.sqrtw = np.repeat(np.sqrt(self.weights), 2)
        ndof = 2*count
        eye = np.eye(ndof)

        adjacency = [set() for _ in range(count)]
        for aa, bb in edges:
            adjacency[aa].add(bb); adjacency[bb].add(aa)

        def angle_between(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na <= 1e-30 or nb <= 1e-30:
                return 180.
            return math.degrees(math.acos(float(np.clip(np.dot(a, b)/(na*nb), -1., 1.))))

        def piece_chain(name):
            nodes = [i for i in range(count) if self.pieces[i] == name]
            node_set = set(nodes)
            ends = [i for i in nodes if len(adjacency[i] & node_set) == 1]
            if len(ends) != 2:
                return []
            path=[ends[0]]; previous=None; current=ends[0]
            while current != ends[1]:
                nxt=list((adjacency[current]&node_set)-({previous} if previous is not None else set()))
                if len(nxt)!=1:
                    return []
                previous,current=current,nxt[0]
                path.append(current)
                if len(path)>len(nodes)+1:
                    return []
            return path

        def nearest_piece_node(name, point):
            ids=np.where(self.pieces==name)[0]
            if not len(ids):
                return None
            d=np.linalg.norm(self.xy[ids]-np.asarray(point,float),axis=1)
            return int(ids[int(np.argmin(d))])

        def path_between(chain, aa, bb):
            if aa not in chain or bb not in chain:
                return []
            ia, ib=chain.index(aa), chain.index(bb)
            return chain[ia:ib+1] if ia<=ib else list(reversed(chain[ib:ia+1]))

        def endpoint_mapping(name, chain, node, point, source_index, source_count):
            k=chain.index(node)
            neighbors=chain[max(0,k-1):k]+chain[k+1:k+2]
            spacing=max([float(np.linalg.norm(self.xy[node]-self.xy[j])) for j in neighbors] or [0.])
            return dict(piece=name,node=int(node),source_index=int(source_index),
                        source_xy=np.asarray(point,float).tolist(),mesh_xy=self.xy[node].tolist(),
                        error_mm=float(np.linalg.norm(self.xy[node]-point)),
                        adjacent_edge_mm=spacing,piece_end=source_index in (0,source_count-1))

        chains={str(name): piece_chain(name) for name in sorted(set(self.pieces))}
        perimeter=sum(np.linalg.norm(self.xy[bb]-self.xy[aa]) for aa,bb in edges)
        edge_lengths=[np.linalg.norm(self.xy[bb]-self.xy[aa]) for aa,bb in edges]
        typical_edge=float(np.median(edge_lengths)) if edge_lengths else 1.

        walls=[]
        bend_zones=[]
        source_wall_layouts={}
        source='mesh_straight_runs'
        if physical_segments:
            source='builtup_segments.csv'
            for raw_name, segments in sorted(physical_segments.items()):
                name=str(raw_name)
                if name not in chains or not chains[name]:
                    continue
                segs=[np.asarray(row,float) for row in segments]
                if not segs:
                    continue
                points=np.vstack([segs[0][0:2]]+[seg[2:4] for seg in segs])
                layout=physical_wall_layout(points, radius_fraction=bend_radius_fraction)
                source_wall_layouts[name]=dict(
                    walls=[dict(start_source=a,end_source=b,start_xy=points[a].tolist(),
                                end_xy=points[b].tolist(),length_mm=float(layout['arclength'][b]-layout['arclength'][a]))
                           for a,b in layout['walls']],
                    curvature_threshold_per_mm=layout['curvature_threshold_per_mm'])
                bend_zones.extend(dict(piece=name,start_xy=points[a].tolist(),end_xy=points[b].tolist(),
                                       length_mm=float(layout['arclength'][b]-layout['arclength'][a]))
                                  for a,b in layout['bends'])
                for a,b in layout['walls']:
                    a0,a1=nearest_piece_node(name,points[a]),nearest_piece_node(name,points[b])
                    path=path_between(chains[name],a0,a1)
                    if len(path)>=3:
                        walls.append(dict(piece=name,path=path,
                            source_length=float(layout['arclength'][b]-layout['arclength'][a]),
                            endpoint_mapping=[
                                endpoint_mapping(name,chains[name],a0,points[a],a,len(points)),
                                endpoint_mapping(name,chains[name],a1,points[b],b,len(points))],
                            mapping_error_mm=max(float(np.linalg.norm(self.xy[a0]-points[a])),
                                                 float(np.linalg.norm(self.xy[a1]-points[b])))))

        else:
            for name,chain in chains.items():
                if len(chain)<3:
                    continue
                vectors=np.diff(self.xy[chain],axis=0)
                runs=[]; start=0; reference=vectors[0]
                for j in range(1,len(vectors)):
                    if angle_between(reference,vectors[j])>wall_angle_deg:
                        runs.append((start,j)); start=j; reference=vectors[j]
                runs.append((start,len(vectors)))
                chain_length=float(np.sum(np.linalg.norm(vectors,axis=1)))
                min_len=max(3.*typical_edge,.04*chain_length)
                for lo,hi in runs:
                    path=chain[lo:hi+1]
                    length=float(np.sum(np.linalg.norm(np.diff(self.xy[path],axis=0),axis=1)))
                    if length>=min_len and len(path)>=3:
                        walls.append(dict(piece=name,path=path,source_length=length))

        unique=[]; seen=set()
        for wall in walls:
            path=wall['path']
            key=(wall['piece'],min(path[0],path[-1]),max(path[0],path[-1]))
            if key in seen:
                continue
            chord=self.xy[path[-1]]-self.xy[path[0]]
            clen=np.linalg.norm(chord)
            if clen<=1e-30:
                continue
            delta=self.xy[path]-self.xy[path[0]]
            deviation=np.abs(delta[:,0]*chord[1]-delta[:,1]*chord[0])/clen
            if not physical_segments and np.max(deviation)>max(.02*clen,.25*typical_edge):
                continue
            seen.add(key); unique.append(wall)
        walls=unique

        fold_nodes=set()
        for wall in walls:
            fold_nodes.update((wall['path'][0],wall['path'][-1]))
        for name,chain in chains.items():
            if chain:
                fold_nodes.update((chain[0],chain[-1]))
        fold_nodes=sorted(fold_nodes)

        pwall=np.zeros((ndof,ndof))
        local_rows=set()
        wall_lengths=[]
        wall_curvature_rows=[]
        for wall in walls:
            path=wall['path']
            pts=self.xy[path]
            aa,bb,normals,dist=wall_kinematics(pts)
            length=float(dist[-1]); wall_lengths.append(length)
            scalar=np.zeros((len(path),ndof))
            for j,node in enumerate(path[1:-1],1):
                if node in local_rows:
                    continue
                normal=normals[j]
                scalar[j,2*node:2*node+2]=normal
                scalar[j,2*path[0]:2*path[0]+2]-=normal@aa[j]
                scalar[j,2*path[-1]:2*path[-1]+2]-=normal@bb[j]
                pwall[2*node:2*node+2,:]=normal[:,None]*scalar[j]
                local_rows.add(node)
            # Curvature of the rigid-corrected normal residual, not of the
            # displacement of the undeformed curved reference shape.
            for j in range(1,len(path)-1):
                h0=dist[j]-dist[j-1]; h1=dist[j+1]-dist[j]
                coeff=np.array([2./(h0*(h0+h1)),-2./(h0*h1),2./(h1*(h0+h1))])
                row=coeff@scalar[j-1:j+2]
                wall_curvature_rows.append(row*length*length*np.sqrt((h0+h1)/(2.*length)))
        self.pwalllocal=pwall
        self.curvature_matrix=np.vstack(wall_curvature_rows) if wall_curvature_rows else np.zeros((0,ndof))

        centered=self.xy-np.average(self.xy,axis=0,weights=self.weights)
        rigid=np.zeros((count,2,3))
        rigid[:,0,0]=1.; rigid[:,1,1]=1.
        rigid[:,0,2]=-centered[:,1]; rigid[:,1,2]=centered[:,0]
        rigid_flat=rigid.reshape(-1,3)
        self.qglobal=orth(rigid_flat*self.sqrtw[:,None])

        def rigid_fit_operator(nodes, output_mask=None):
            nodes=list(nodes)
            rows=np.array([2*i+j for i in nodes for j in (0,1)],dtype=int)
            if len(rows)<4:
                return np.zeros((ndof,ndof)),0
            design=rigid_flat[rows]; ws=self.sqrtw[rows]
            weighted=design*ws[:,None]
            rank=int(np.linalg.matrix_rank(weighted,tol=max(weighted.shape)*np.finfo(float).eps*
                                           max(np.linalg.norm(weighted,2),1.)))
            if rank<3:
                return np.zeros((ndof,ndof)),rank
            pinv=np.linalg.pinv(weighted,rcond=1e-12)
            op=np.zeros((ndof,ndof)); op[:,rows]=rigid_flat@pinv@np.diag(ws)
            if output_mask is not None:
                keep=np.repeat(np.asarray(output_mask,dtype=bool),2); op[~keep,:]=0.
            return op,rank

        residual_after_local=eye-self.pwalllocal
        fold_rows=np.array([2*i+j for i in fold_nodes for j in (0,1)],dtype=int)
        if not compute_bases and np.any(self.pwalllocal[fold_rows]):
            raise ValueError('Overlapping physical walls place local residuals on fold anchors')
        self.pglobal,global_fold_rank=rigid_fit_operator(fold_nodes)
        if compute_bases:
            self.pglobal=self.pglobal@residual_after_local
        after_global=residual_after_local-self.pglobal

        piece_total=np.zeros((ndof,ndof)); piece_fold_ranks={}
        for name in sorted(set(self.pieces)):
            nodes=[i for i in fold_nodes if self.pieces[i]==name]
            mask=self.pieces==name
            op,rank=rigid_fit_operator(nodes,mask)
            piece_fold_ranks[str(name)]=rank
            piece_total+=op
        self.passembly=(piece_total@after_global if compute_bases else
                        piece_total-piece_total[:,fold_rows]@self.pglobal[fold_rows])
        after_assembly=after_global-self.passembly

        fold_ids={node:i for i,node in enumerate(fold_nodes)}
        gather=np.zeros((2*len(fold_nodes),ndof))
        for k,node in enumerate(fold_nodes):
            gather[2*k,2*node]=1.; gather[2*k+1,2*node+1]=1.

        interpolation=np.zeros((ndof,2*len(fold_nodes)))
        for node,k in fold_ids.items():
            interpolation[2*node:2*node+2,2*k:2*k+2]=np.eye(2)
        for name,chain in chains.items():
            marks=sorted((chain.index(n),n) for n in fold_nodes if n in chain)
            for (_,aa),(_,bb) in zip(marks,marks[1:]):
                path=path_between(chain,aa,bb)
                if len(path)<2:
                    continue
                left,right,unused_normal,unused_dist=wall_kinematics(self.xy[path])
                for j,node in enumerate(path):
                    interpolation[2*node:2*node+2,2*fold_ids[aa]:2*fold_ids[aa]+2]=left[j]
                    interpolation[2*node:2*node+2,2*fold_ids[bb]:2*fold_ids[bb]+2]=right[j]

        constraints=[]
        for wall in walls:
            aa,bb=wall['path'][0],wall['path'][-1]
            if aa not in fold_ids or bb not in fold_ids:
                continue
            chord=self.xy[bb]-self.xy[aa]; length=np.linalg.norm(chord)
            if length<=1e-30: continue
            tangent=chord/length
            row=np.zeros(2*len(fold_nodes))
            row[2*fold_ids[aa]:2*fold_ids[aa]+2]-=tangent
            row[2*fold_ids[bb]:2*fold_ids[bb]+2]+=tangent
            constraints.append(row)
        fold_weights=np.repeat(np.sqrt(self.weights[fold_nodes]),2) if fold_nodes else np.empty(0)
        if constraints:
            c=np.vstack(constraints); cw=c/fold_weights[None,:]
            unused_u,singular,vt=np.linalg.svd(cw,full_matrices=True)
            crank=int(np.sum(singular>(singular[0]*1e-10 if len(singular) else 0.)))
            nullw=vt[crank:].T
            pd=(nullw@nullw.T)*fold_weights[None,:]/fold_weights[:,None]
            po=np.eye(2*len(fold_nodes))-pd
        else:
            crank=0; pd=np.eye(2*len(fold_nodes)); po=np.zeros_like(pd)

        fold_residual=gather@after_assembly if compute_bases else after_assembly[fold_rows]
        self.pdist=interpolation@pd@fold_residual
        coarse_other=interpolation@po@fold_residual if compute_bases else None
        remainder=eye-self.pwalllocal-self.pglobal-self.passembly-self.pdist
        if compute_bases: remainder-=coarse_other
        # Any non-chord, non-fold residual is membrane/shear/corner-zone motion,
        # not plate bending. Keep it out of L and report it as Other.
        self.pother=coarse_other+remainder if compute_bases else remainder
        self.plocal=self.pwalllocal

        to_weighted=np.diag(self.sqrtw); from_weighted=np.diag(1./self.sqrtw)
        # Fast suggestion uses the displacement operators directly. Dense SVD
        # bases/ranks are required by the audit, but add no screening information.
        if compute_bases:
            self.qassembly=orth(to_weighted@self.passembly@from_weighted)
            self.qdist=orth(to_weighted@self.pdist@from_weighted)
            self.qother=orth(to_weighted@self.pother@from_weighted)
            self.qlocal=orth(to_weighted@self.plocal@from_weighted)
        else:
            self.qassembly=self.qdist=self.qother=self.qlocal=None

        piece_ranks_ok=all(rank>=3 for rank in piece_fold_ranks.values())
        self.supported=bool(len(walls)>=2 and len(fold_nodes)>=2 and global_fold_rank>=3 and
                            piece_ranks_ok and all(chains.values()) and len(local_rows)>0)
        self.flat_fraction=float(sum(wall_lengths)/max(perimeter,1e-30))
        self.metadata=dict(
            wall_method="compact_source_bends_rigid_exact_v2" if physical_segments else "mesh_straight_runs",
            bend_radius_fraction=bend_radius_fraction, bend_zones=bend_zones,
            source_wall_layouts=source_wall_layouts,
            maximum_wall_mapping_error_mm=max([w.get("mapping_error_mm",0.) for w in walls] or [0.]),
            wall_endpoint_mapping=[item for w in walls for item in w.get('endpoint_mapping',[])],
            fold_nodes=fold_nodes, physical_wall_count=len(walls),
            physical_walls=[dict(piece=w['piece'],start=w['path'][0],end=w['path'][-1],
                                 node_count=len(w['path']),length_mm=float(np.sum(np.linalg.norm(
                                 np.diff(self.xy[w['path']],axis=0),axis=1)))) for w in walls],
            wall_source=source, wall_angle_deg=wall_angle_deg,
            flat_panel_length_fraction=self.flat_fraction,
            physical_wall_coverage_fraction=self.flat_fraction,
            geometry_supported=self.supported,
            curved_panel_proxy=(not physical_segments or self.flat_fraction<.6),
            global_anchor_rank=global_fold_rank, piece_anchor_ranks=piece_fold_ranks,
            global_rank=self.qglobal.shape[1],
            assembly_rank=self.qassembly.shape[1] if compute_bases else None,
            coarse_deformation_rank=self.qdist.shape[1] if compute_bases else None,
            other_extension_rank=self.qother.shape[1] if compute_bases else None,
            local_rank=self.qlocal.shape[1] if compute_bases else None,
            wall_bending_rank=int(np.linalg.matrix_rank(self.pwalllocal)) if compute_bases else len(local_rows),
            panel_extension_constraint_rank=crank,
            maximum_panel_arc_length=float(max(wall_lengths) if wall_lengths else 0.),
            split_definition='normal wall bending relative to rigid-exact moving curved reference first; fold-driven G/A/O/D; tangential and compact-bend residual assigned to Other')

    def _weighted_components(self, coefficients):
        y=np.asarray(coefficients,dtype=float)
        original_shape=y.shape
        y=y.reshape(-1,len(self.sqrtw)); x=y/self.sqrtw[None,:]
        def apply(operator):
            return (x@operator.T)*self.sqrtw[None,:]
        parts=dict(G=apply(self.pglobal),A=apply(self.passembly),O=apply(self.pother),
                   D=apply(self.pdist),L=apply(self.plocal))
        for key in parts:
            parts[key]=parts[key].reshape(original_shape)
        return parts

    def component_diagnostics_weighted(self, weighted_coefficients):
        parts=self._weighted_components(weighted_coefficients)
        norms={k:float(np.sum(v*v)) for k,v in parts.items()}
        ldg=norms['L']+norms['D']+norms['G']; all_self=ldg+norms['A']+norms['O']
        # Pure Other/Assembly motion can leave round-off in L/D/G. Do not
        # normalize that negligible remainder into apparently large shares.
        if ldg <= 1e-24*all_self:
            for key in ('L', 'D', 'G'):
                norms[key]=0.
            ldg=0.
        reconstruction=sum(parts.values()); y=np.asarray(weighted_coefficients,dtype=float)
        x=(y.reshape(-1,len(self.sqrtw))/self.sqrtw[None,:])
        curvature=float(np.sum((x@self.curvature_matrix.T)**2)) if self.curvature_matrix.size else 0.
        return dict(
            global_percent=100*norms['G']/max(ldg,1e-250),
            distortional_percent=100*norms['D']/max(ldg,1e-250),
            local_percent=100*norms['L']/max(ldg,1e-250),
            assembly_percent=100*norms['A']/max(all_self,1e-250),
            other_percent=100*norms['O']/max(all_self,1e-250),
            wall_curvature_index=curvature/max(float(np.sum(x*x)),1e-250),
            reconstruction_relative_error=float(np.linalg.norm(y-reconstruction)/max(np.linalg.norm(y),1e-250)),
            component_norm_sum_over_input=float(all_self/max(float(np.sum(y*y)),1e-250)))

    def component_diagnostics(self, coefficients):
        x=np.asarray(coefficients,dtype=float)
        weighted=(x.reshape(-1,len(self.sqrtw))*self.sqrtw[None,:]).reshape(x.shape)
        return self.component_diagnostics_weighted(weighted)

    def shares(self, coefficients):
        d=self.component_diagnostics(coefficients)
        return [d['global_percent']/100.,d['distortional_percent']/100.,d['local_percent']/100.]

    def audit_shares(self, weighted_coefficients):
        d=self.component_diagnostics_weighted(weighted_coefficients)
        return [d['local_percent'],d['distortional_percent'],d['global_percent']]

    def audit_components(self, weighted_coefficients):
        p=self._weighted_components(weighted_coefficients)
        return [p['L'],p['D'],p['G'],p['A'],p['O']]


def family_label(shares, supported, threshold=.9, poor_fit=False,
                 assembly_percent=0., max_assembly_percent=25.,
                 other_percent=0., max_other_percent=25.):
    if poor_fit:
        return 'Unresolved'
    if assembly_percent >= max_assembly_percent:
        return 'Assembly-like'
    if other_percent >= max_other_percent:
        return 'Other-like'
    if sum(shares) <= 0:
        return 'Unresolved'
    if shares[0] >= threshold:
        return 'Global-like'
    if not supported:
        return 'Unresolved'
    if shares[1] >= threshold:
        return 'Distortional-like'
    if shares[2] >= threshold:
        return 'Local-like'
    return 'Mixed'

def sample_envelope(rows, length, value_key):
    grouped = {}
    for row in rows:
        if row.get('dominant_halfwaves') and row.get(value_key) and row[value_key] > 0:
            grouped.setdefault(row['dominant_halfwaves'], []).append(row)
    result = []
    for n, items in sorted(grouped.items()):
        best = min(items, key=lambda r: r[value_key])
        filtered = [r for r in items if r['quality_ok']]
        clean = min(filtered, key=lambda r: r[value_key]) if filtered else None
        result.append(dict(halfwaves=n, half_wavelength_mm=length/n, sample_count=len(items),
            minimum_value=best[value_key], minimum_mode=best['mode'],
            filtered_minimum_value=clean[value_key] if clean else None,
            filtered_minimum_mode=clean['mode'] if clean else None))
    return result


def envelope_line(rows, key):
    x, y, previous = [], [], None
    for row in rows:
        if previous is not None and row['halfwaves'] != previous+1:
            x.append(float('nan'))
            y.append(float('nan'))
        x.append(row['half_wavelength_mm'])
        y.append(row[key] if row[key] is not None else float('nan'))
        previous = row['halfwaves']
    return x, y


def load_base(module_path):
    spec = importlib.util.spec_from_file_location('enhanced_base_processor', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_results(report_path):
    report_path = os.path.abspath(report_path)
    if not report_path.endswith('_report.json'):
        raise ValueError('Input must be the existing *_modal_wavelengths_report.json')
    prefix = report_path[:-len('_report.json')]
    with open(report_path) as stream:
        metadata = json.load(stream)
    with open(prefix+'.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    integer = ('mode', 'dominant_halfwaves', 'max_resolved_halfwaves')
    floats = ('eigenvalue', 'critical_stress_MPa', 'half_wavelength_mm', 'dominant_share',
              'relative_fit_error', 'end_displacement_ratio', 'transverse_displacement_share')
    for row in rows:
        for key in integer:
            row[key] = int(row[key]) if row[key] else None
        for key in floats:
            row[key] = float(row[key]) if row[key] else None
    rows.sort(key=lambda r: r['mode'])
    modes = [r['mode'] for r in rows]
    if len(set(modes)) != len(modes) or len(rows) != metadata['processed_modes']:
        raise ValueError('Mode CSV and report disagree, or contain duplicate modes')
    spectra = np.full((len(rows), metadata['max_resolved_halfwaves']), np.nan)
    index = {mode: i for i, mode in enumerate(modes)}
    with open(prefix+'_spectra.csv', newline='') as stream:
        for row in csv.DictReader(stream):
            i, j = index[int(row['mode'])], int(row['halfwaves'])-1
            if not math.isnan(spectra[i, j]):
                raise ValueError('Duplicate spectrum entry')
            spectra[i, j] = float(row['displacement_coefficient_share'])
    if not np.all(np.isfinite(spectra)) or np.any(spectra < 0):
        raise ValueError('Incomplete or invalid spectrum CSV')
    for i, row in enumerate(rows):
        if row['dominant_halfwaves']:
            if not np.isclose(spectra[i].sum(), 1., atol=1e-6):
                raise ValueError('Spectrum shares do not sum to one')
            if not np.isclose(spectra[i, row['dominant_halfwaves']-1], row['dominant_share']):
                raise ValueError('Spectrum and mode CSV disagree')
    return metadata, rows, spectra, prefix


def load_physical_segments(report_dir):
    """Read physical wall geometry persisted by the one-command builder."""
    candidates=[name for name in os.listdir(report_dir) if name.endswith('_build.json')]
    if len(candidates)!=1:
        return None
    try:
        with open(os.path.join(report_dir,candidates[0])) as stream:
            build=json.load(stream)
    except (OSError,ValueError):
        return None
    return (build.get('source_inputs') or {}).get('section_segments')


def section_diagnostics(base, metadata, rows, spectra, report_dir, wall_angle_deg, family_threshold, max_assembly_percent, max_other_percent):
    from odbAccess import openOdb
    odb_path = os.path.join(report_dir, os.path.basename(metadata['odb']))
    if not os.path.isfile(odb_path):
        odb_path = metadata['odb']
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        axis = 'xyz'.index(metadata['axis'])
        transverse = [i for i in range(3) if i != axis]
        names = [item['instance'] for item in metadata['instances']]
        tracks, meshmeta, keys, areas, origin, length, tolerance = base.collect_tracks(
            odb, names, axis, metadata['settings']['coord_tolerance'])
        if not np.isclose(length, metadata['length_mm']):
            raise ValueError('ODB length differs from the previous report')
        groups, cap = base.build_groups(tracks, origin, length, metadata['max_resolved_halfwaves'])
        if cap != spectra.shape[1]:
            raise ValueError('ODB mesh/spectrum resolution mismatch; regenerate base outputs')
        lookup = {key: i for i, key in enumerate(keys)}
        tables = base.label_tables(keys)
        coordinates = {(name, n.label): np.asarray(n.coordinates, dtype=float)
                       for name in names for n in odb.rootAssembly.instances[name].nodes}
        track_index = {}
        for i, track in enumerate(tracks):
            for index in track['indices']:
                track_index[index] = i
        xy = np.array([coordinates[keys[t['indices'][0]]][transverse] for t in tracks])
        edges = set()
        for name in names:
            for element in odb.rootAssembly.instances[name].elements:
                nodes = base.shell_corners(element)
                for a, b in zip(nodes, nodes[1:]+nodes[:1]):
                    ka, kb = (name, a), (name, b)
                    if ka not in lookup or kb not in lookup:
                        continue
                    ia, ib = track_index[lookup[ka]], track_index[lookup[kb]]
                    if ia != ib and abs(coordinates[ka][axis]-coordinates[kb][axis]) <= tolerance:
                        edges.add(tuple(sorted((ia, ib))))
        edges = sorted(edges)
        physical_segments=load_physical_segments(report_dir)
        fit = SectionProjector(xy, edges, [t['instance'] for t in tracks],
                               [t['weight'] for t in tracks],
                               physical_segments=physical_segments, wall_angle_deg=wall_angle_deg)
        group_tracks = [[track_index[int(index)] for index in group['indices'][0]] for group in groups]
        frames = {}
        for frame in odb.steps[metadata['step']].frames:
            item = base.frame_eigen(frame)
            if item:
                if item[0] in frames:
                    raise ValueError('Duplicate ODB mode numbers')
                frames[item[0]] = (item[1], frame)
        shapes = []
        for i, row in enumerate(rows):
            eigenvalue, frame = frames[row['mode']]
            if not np.isclose(eigenvalue, row['eigenvalue'], rtol=1e-7, atol=1e-8):
                raise ValueError('ODB eigenvalue differs from CSV at mode %d' % row['mode'])
            displacement = base.read_displacements(frame, lookup, len(keys), tables)
            coefficients = np.zeros((cap, len(tracks), 2))
            for group, indices in zip(groups, group_tracks):
                values = displacement[group['indices']][:, :, transverse]
                projected = group['fit'].project.dot(values[group['fit'].order].reshape(len(values), -1))
                coefficients[:, indices, :] = projected.reshape(cap, len(indices), 2)
            # Match original spectra, not just mode count, before combining old/new data.
            power = np.sum(coefficients**2*np.array([t['weight'] for t in tracks])[None, :, None], axis=(1, 2))
            if power.sum() > 1e-300 and not np.allclose(power/power.sum(), spectra[i], atol=1e-6, rtol=1e-5):
                raise ValueError('ODB mode shape differs from original spectrum at mode %d' % row['mode'])
            shares = fit.shares(coefficients)
            split = fit.component_diagnostics(coefficients)
            bad = (row['relative_fit_error'] is None or
                   row['relative_fit_error'] > metadata['settings']['max_relative_error'] or
                   row['end_displacement_ratio'] > .05 or
                   row['transverse_displacement_share'] < 1e-8)
            row.update(global_proxy_share=shares[0], distortional_proxy_share=shares[1],
                       local_proxy_share=shares[2], assembly_proxy_share=split['assembly_percent']/100.,
                       other_proxy_share=split['other_percent']/100.,
                       wall_curvature_index=split.get('wall_curvature_index'),
                       family=family_label(shares, fit.supported, family_threshold, bad,
                                           split['assembly_percent'], max_assembly_percent,
                                           split['other_percent'], max_other_percent),
                       classification_basis='heuristic_transverse_displacement',
                       geometry_proxy_supported=fit.supported)
            n = row['dominant_halfwaves']
            shapes.append(coefficients[n-1].tolist() if n else np.zeros((len(tracks), 2)).tolist())
            if (i+1) % 50 == 0 or i == len(rows)-1:
                print('Enhanced section diagnostics: %d/%d modes' % (i+1, len(rows)))
                sys.stdout.flush()
        return dict(xy=xy.tolist(), edges=edges, pieces=[t['instance'] for t in tracks],
                    dominant_harmonic_shapes=shapes, diagnostics=fit.metadata, odb=odb_path)
    finally:
        odb.close()


def make_plots(prefix, metadata, rows, spectra, envelope, value_key, near_fraction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    length, cap = metadata['length_mm'], spectra.shape[1]
    limit, caution = length/cap, length/(cap*near_fraction)
    ylabel = 'Critical stress (MPa)' if value_key == 'critical_stress_MPa' else 'Eigenvalue (load multiplier)'
    def wavelength_axis(ax):
        ax.set_xscale('log')
        ax.set_xlim(limit*.65, length*1.15)
        ax.axvspan(limit*.65, limit, color='#dddddd', hatch='///', alpha=.5)
        ax.axvspan(limit, caution, color='#e6ad30', alpha=.12)
        ax.axvline(limit, color='#b13333', linestyle=':', linewidth=1.)
        ax.set_xlabel('Half-wavelength (mm); hatched = outside resolved range')
        ax.grid(True, which='both', alpha=.2)
    fig, ax = plt.subplots(figsize=(12, 7))
    for family, color in COLORS.items():
        subset = [r for r in rows if r['family'] == family and r['half_wavelength_mm'] and r[value_key] > 0]
        if subset:
            ax.scatter([r['half_wavelength_mm'] for r in subset], [r[value_key] for r in subset],
                s=[25+125*r['dominant_share'] for r in subset], c=color, alpha=.8,
                label='%s (%d)' % (family, len(subset)), edgecolors='white', linewidths=.4)
    limited = [r for r in rows if r['half_wavelength_mm'] and r[value_key] > 0 and not r['quality_ok']]
    if limited:
        ax.scatter([r['half_wavelength_mm'] for r in limited], [r[value_key] for r in limited],
                   marker='x', s=20, color='#303030', linewidths=.65, label='Spectral/fit/resolution flag')
    for family in COLORS:
        selected = [r for r in rows if r['family'] == family and r['half_wavelength_mm'] and r[value_key] > 0]
        if selected:
            r = min(selected, key=lambda item: item[value_key])
            ax.annotate('m%d' % r['mode'], (r['half_wavelength_mm'], r[value_key]),
                        xytext=(5, -12), textcoords='offset points', fontsize=8)
    wavelength_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title('Fixed-length modal samples | L = %g mm\nHeuristic section families; marker size = dominant harmonic share' % length)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.text(.06, .015, 'Family shares are displacement proxies, not strain energies. No mesh-convergence claim. Not a classical signature curve.', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(prefix+'_modes.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    half = length/np.arange(cap, 0, -1)
    if cap > 1:
        centers = np.sqrt(half[:-1]*half[1:])
        bounds = np.r_[half[0]**2/centers[0], centers, half[-1]**2/centers[-1]]
    else:
        bounds = np.array([half[0]*.9, half[0]*1.1])
    picture = ax.pcolormesh(bounds, np.arange(len(rows)+1)-.5, spectra[:, ::-1],
                           shading='flat', cmap='viridis', norm=PowerNorm(gamma=.45, vmin=0, vmax=1), rasterized=True)
    ax.scatter([r['half_wavelength_mm'] or np.nan for r in rows], np.arange(len(rows)),
               s=9, facecolors='none', edgecolors='white', linewidths=.5)
    wavelength_axis(ax)
    ax.set_ylim(len(rows)-.5, -.5)
    ticks = np.unique(np.linspace(0, len(rows)-1, min(12, len(rows))).astype(int))
    ax.set_yticks(ticks)
    ax.set_yticklabels([rows[i]['mode'] for i in ticks])
    ax.set_ylabel('Mode number')
    ax.set_title('Longitudinal spectrum of every mode\nWhite circles: dominant harmonic; colors: squared-displacement coefficient shares')
    fig.colorbar(picture, ax=ax, label='Harmonic share', ticks=[0, .05, .1, .25, .5, .75, 1.])
    fig.tight_layout()
    fig.savefig(prefix+'_spectra.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for key, style, label in [('minimum_value', 'o--', 'Minimum of available dominant-half-wave samples'),
            ('filtered_minimum_value', 's-', 'Minimum after spectral/fit/resolution filtering')]:
        x, y = envelope_line(envelope, key)
        ax.plot(x, y, style, markersize=4, linewidth=1., label=label)
    wavelength_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title('Discrete sample lower envelope | NOT a classical signature curve\nGaps remain where no dominant-half-wave sample is available')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(prefix+'_envelope.png', dpi=180)
    plt.close(fig)


def write_html(prefix, metadata, rows, spectra, geometry, value_key):
    images = {}
    for name in ('modes', 'spectra', 'envelope'):
        with open(prefix+'_'+name+'.png', 'rb') as stream:
            images[name] = base64.b64encode(stream.read()).decode('ascii')
    payload = json.dumps(dict(rows=rows, spectra=spectra.tolist(), geometry=geometry,
        length=metadata['length_mm'], valueKey=value_key, colors=COLORS), ensure_ascii=True, allow_nan=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Enhanced buckling mode report</title>
<style>body{font:15px system-ui;margin:25px auto;max-width:1200px;padding:0 20px;color:#203044;background:#f6f8fb}h1,h2{font-weight:650}section{background:white;padding:20px;margin:20px 0;border:1px solid #dde2ea;border-radius:9px}img{width:100%}button,select,input{font:inherit;padding:6px;margin:4px}svg{width:100%;height:auto;background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}tr[data-index]{cursor:pointer}tr[data-index]:hover{background:#edf3ff}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.note{color:#5b6576;font-size:13px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Fixed-length buckling modes</h1><p>Geometric family labels are <b>heuristic</b>. This report is not a classical signature curve.</p>
<p class="note">__LIMITATION__</p>
<section><h2>1. Modal samples</h2><p>Colors identify the proposed section family. Size measures the dominant harmonic share, not statistical confidence. Crosses identify flagged wavelength estimates. Mode numbers shown on the plot are selected family minima.</p><img src="data:image/png;base64,__MODES__"></section>
<section><h2>2. Inspect a mode and its main harmonics</h2><button id="prev">Previous</button><select id="mode"></select><button id="next">Next</button><label>Drawing amplification <input id="amplitude" type="range" min="0" max="30" value="10"></label><div id="details"></div>
<div class="grid"><div><h3>Dominant-harmonic section shape</h3><svg id="shape" viewBox="0 0 500 450"></svg><p class="note">Gray dashed: original section; colored: dominant-harmonic transverse coefficients at arbitrary scale. This is not a physical displacement amplitude or a full multimodal deformation snapshot.</p></div><div><h3>Longitudinal harmonic shares</h3><svg id="spectrum" viewBox="0 0 500 450"></svg><p class="note">All harmonics are shown; the table lists the most important ones. Multiple half-wavelengths belong to the same eigenvalue, not independent buckling solutions.</p></div></div><div id="components"></div></section>
<section><h2>3. Spectrum map</h2><img src="data:image/png;base64,__SPECTRA__"></section>
<section><h2>4. Lower envelope of available samples</h2><p>Only positive-eigenvalue dominant-half-wave samples are used. Secondary components are not treated as independent solutions. Lines do not bridge missing harmonics. Filtering is not mesh-convergence verification.</p><img src="data:image/png;base64,__ENVELOPE__"></section>
<section><h2>5. Mode table</h2><label>Family <select id="family"><option>All</option></select></label><div style="max-height:600px;overflow:auto"><table><thead><tr><th>Mode</th><th>Eigenvalue</th><th>Half-wave mm</th><th>Dominant %</th><th>Family (heuristic)</th><th>G / D / L %</th><th>Flags</th></tr></thead><tbody id="table"></tbody></table></div></section>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
function svg(tag,attrs,parent,text){let e=document.createElementNS(NS,tag);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;parent.appendChild(e);return e}
function fmt(x,n=2){return x===null?'--':Number(x).toFixed(n)}
D.rows.forEach((r,i)=>{let o=new Option('Mode '+r.mode+' | '+r.family,i);$('mode').add(o)});
Object.keys(D.colors).forEach(f=>$('family').add(new Option(f,f)));
function render(){let i=+$('mode').value,r=D.rows[i], color=D.colors[r.family], shares=[r.global_proxy_share,r.distortional_proxy_share,r.local_proxy_share];
$('details').textContent='Eigenvalue '+fmt(r.eigenvalue)+' | Half-wave '+fmt(r.half_wavelength_mm)+' mm | '+r.family+' | G/D/L proxies '+shares.map(v=>fmt(100*v,1)+'%').join(' / ')+' | Assembly '+fmt(100*(r.assembly_proxy_share||0),1)+'% | Other '+fmt(100*(r.other_proxy_share||0),1)+'% | Flags: '+r.enhanced_flags;
let xy=D.geometry.xy, u=D.geometry.dominant_harmonic_shapes[i], mins=[0,1].map(k=>Math.min(...xy.map(p=>p[k]))), maxs=[0,1].map(k=>Math.max(...xy.map(p=>p[k]))), span=Math.max(maxs[0]-mins[0],maxs[1]-mins[1],1), center=mins.map((v,k)=>(v+maxs[k])/2), umax=Math.max(...u.map(v=>Math.hypot(...v)),1e-30), gain=span*(+$('amplitude').value/100)/umax;
let map=p=>[250+(p[0]-center[0])*330/span,225-(p[1]-center[1])*330/span], deformed=xy.map((p,j)=>p.map((v,k)=>v+gain*u[j][k]));$('shape').replaceChildren();
for(let [a,b] of D.geometry.edges){for(let [points,stroke,dash] of [[xy,'#9ca7b2','4 3'],[deformed,color,'']]){let p=map(points[a]),q=map(points[b]);svg('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:stroke,'stroke-width':2,'stroke-dasharray':dash},$('shape'))}}
for(let j of (D.geometry.diagnostics.fold_nodes||D.geometry.diagnostics.anchor_nodes||[])){let p=map(xy[j]);svg('circle',{cx:p[0],cy:p[1],r:2,fill:'#45556a'},$('shape'))}
$('spectrum').replaceChildren();let s=D.spectra[i],w=420/s.length;
svg('line',{x1:50,y1:390,x2:475,y2:390,stroke:'#333'},$('spectrum'));
s.forEach((v,j)=>{let e=svg('rect',{x:50+j*w,y:390-v*340,width:Math.max(w-.7,.3),height:v*340,fill:color},$('spectrum'));svg('title',{},e,'n='+(j+1)+'; half-wave='+fmt(D.length/(j+1))+' mm; share='+fmt(v*100)+'%');if(j===0||(j+1)%5===0)svg('text',{x:50+(j+.5)*w,y:410,'text-anchor':'middle','font-size':11},$('spectrum'),j+1)});
for(let v of [0,.25,.5,.75,1])svg('text',{x:43,y:394-v*340,'text-anchor':'end','font-size':11},$('spectrum'),Math.round(v*100)+'%');
svg('text',{x:250,y:439,'text-anchor':'middle','font-size':12},$('spectrum'),'Number of longitudinal half-waves n');
$('components').replaceChildren();let p=document.createElement('p');p.textContent='Top components: '+r.top_components.map(c=>'n='+c.halfwaves+', h='+fmt(c.half_wavelength_mm)+' mm, '+fmt(c.share*100,1)+'%').join(' | ')+'; retained share '+fmt(r.top_components_share*100,1)+'%';$('components').appendChild(p)}
function table(){let target=$('table');target.replaceChildren();D.rows.forEach((r,i)=>{if($('family').value!=='All'&&r.family!==$('family').value)return;let tr=document.createElement('tr');tr.dataset.index=i;for(let v of [r.mode,fmt(r.eigenvalue),fmt(r.half_wavelength_mm),fmt(100*r.dominant_share,1),r.family,[r.global_proxy_share,r.distortional_proxy_share,r.local_proxy_share].map(x=>fmt(100*x,1)).join(' / '),r.enhanced_flags]){let td=document.createElement('td');td.textContent=v;tr.appendChild(td)}tr.onclick=()=>{$('mode').value=i;render();$('mode').scrollIntoView({behavior:'smooth'})};target.appendChild(tr)})}
$('mode').onchange=render;$('amplitude').oninput=render;$('family').onchange=table;
$('prev').onclick=()=>{$('mode').value=Math.max(0,+$('mode').value-1);render()};$('next').onclick=()=>{$('mode').value=Math.min(D.rows.length-1,+$('mode').value+1);render()};render();table();
</script></html>'''
    for key, value in {'__LIMITATION__': html.escape(LIMITATION), '__DATA__': payload,
                       '__MODES__': images['modes'], '__SPECTRA__': images['spectra'],
                       '__ENVELOPE__': images['envelope']}.items():
        page = page.replace(key, value)
    with open(prefix+'.html', 'w', encoding='utf-8') as stream:
        stream.write(page)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modal_report')
    parser.add_argument('--output', help='Output prefix; default: base prefix + _enhanced')
    parser.add_argument('--top-components', type=int, default=3)
    parser.add_argument('--corner-angle', type=float, default=15., help='Deprecated compatibility option')
    parser.add_argument('--wall-angle-deg', type=float, default=3.,
                        help='Direction-change threshold for the mesh fallback when source wall geometry is unavailable')
    parser.add_argument('--family-threshold', type=float, default=.9)
    parser.add_argument('--max-assembly-percent', type=float, default=25.,
                        help='Above this self-norm share, report Assembly-like instead of forcing L/D/G')
    parser.add_argument('--max-other-percent', type=float, default=25.,
                        help='Above this extension-like self-norm share, report Other-like instead of forcing L/D/G')
    parser.add_argument('--near-limit-fraction', type=float, default=.8)
    args = parser.parse_args(argv)
    if (args.top_components < 1 or not 0 < args.corner_angle < 180 or not 0 < args.wall_angle_deg < 45 or
            not .5 < args.family_threshold <= 1 or not 0 < args.near_limit_fraction <= 1 or
            not 0 < args.max_assembly_percent < 100 or not 0 < args.max_other_percent < 100):
        parser.error('Invalid component count, angle, family threshold or assembly threshold')
    metadata, rows, spectra, base_prefix = load_results(args.modal_report)
    prefix = os.path.abspath(args.output or base_prefix+'_enhanced')
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    base = load_base(os.path.join(SCRIPT_DIR, 'abaqus_modal_wavelengths.py'))
    geometry = section_diagnostics(base, metadata, rows, spectra,
        os.path.dirname(os.path.abspath(args.modal_report)), args.wall_angle_deg, args.family_threshold,
        args.max_assembly_percent, args.max_other_percent)
    components = []
    for i, row in enumerate(rows):
        order = np.argsort(-spectra[i], kind='stable')[:args.top_components]
        top = [dict(halfwaves=int(j)+1, half_wavelength_mm=metadata['length_mm']/(j+1),
                    share=float(spectra[i, j])) for j in order if spectra[i, j] > 1e-12]
        row['top_components'] = top
        row['top_components_share'] = sum(c['share'] for c in top)
        n = row['dominant_halfwaves']
        row['near_resolution_limit'] = bool(n and n >= args.near_limit_fraction*spectra.shape[1])
        flags = [] if row['status'] == 'dominant' else row['status'].split(';')
        if row['near_resolution_limit']:
            flags.append('near_resolution_limit')
        if row['family'] in ('Unresolved', 'Assembly-like', 'Other-like'):
            flags.append('section_family_'+row['family'].lower().replace('-', '_'))
        if geometry['diagnostics']['curved_panel_proxy']:
            flags.append('curved_panel_proxy')
        row['enhanced_flags'] = ';'.join(flags) if flags else 'none'
        row['quality_ok'] = bool(n and row['status'] == 'dominant' and not row['near_resolution_limit'])
        for rank, component in enumerate(top, 1):
            components.append(dict(mode=row['mode'], rank=rank, eigenvalue=row['eigenvalue'],
                critical_stress_MPa=row['critical_stress_MPa'], family=row['family'], **component))
    value_key = 'critical_stress_MPa' if metadata['reference_stress_MPa'] is not None else 'eigenvalue'
    envelope = sample_envelope(rows, metadata['length_mm'], value_key)
    base.write_csv(prefix+'_modes.csv', rows, [key for key in rows[0] if key != 'top_components'])
    base.write_csv(prefix+'_components.csv', components,
        ['mode', 'rank', 'eigenvalue', 'critical_stress_MPa', 'family', 'halfwaves', 'half_wavelength_mm', 'share'])
    base.write_csv(prefix+'_envelope.csv', envelope, ['halfwaves', 'half_wavelength_mm', 'sample_count',
        'minimum_value', 'minimum_mode', 'filtered_minimum_value', 'filtered_minimum_mode'])
    make_plots(prefix, metadata, rows, spectra, envelope, value_key, args.near_limit_fraction)
    write_html(prefix, metadata, rows, spectra, geometry, value_key)
    result = dict(processed_modes=len(rows), source_report=os.path.abspath(args.modal_report),
        output_prefix=prefix, family_counts={family: sum(r['family']==family for r in rows) for family in COLORS},
        classification_limitations=LIMITATION, settings=vars(args), geometry=geometry['diagnostics'],
        minimum_resolved_half_wavelength_mm=metadata['length_mm']/spectra.shape[1],
        caution_half_wavelength_mm=metadata['length_mm']/(args.near_limit_fraction*spectra.shape[1]),
        resolution_note='Four longitudinal intervals per half-wave safeguard; NOT a convergence study.',
        envelope_note='Minima of existing positive dominant-half-wave samples only; not a classical signature curve.',
        section_shape_note='Dominant harmonic coefficients, arbitrarily amplified; not physical deformation amplitude.',
        files=[prefix+suffix for suffix in ('.html', '_modes.png', '_spectra.png', '_envelope.png',
                '_modes.csv', '_components.csv', '_envelope.csv', '_report.json')])
    with open(prefix+'_report.json', 'w') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('ENHANCED COMPLETE: %d modes; %s.html' % (len(rows), prefix))
    return result


if __name__ == '__main__':
    main()
