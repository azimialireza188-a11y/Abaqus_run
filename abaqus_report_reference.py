"""Model-native gross four-piece references and optional legacy Excel reader.

Geometry calculations use the existing shell centre-line integration routines.
Explicit reports remain read-only overrides; no workbook is needed by default.
All capacities use the current column fy, never an auxiliary plate strength.
"""
import hashlib
import math
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile


_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_GROUP = "builtup_4_pieces"
_SHEET = "section_properties_3"
_PROPERTIES = {
    "A": "mm^2",
    "S_x = I / c_max": "mm^3",
    "S_z = I / c_max": "mm^3",
    "Z_x (plastic modulus)": "mm^3",
    "Z_z (plastic modulus)": "mm^3",
    "M_y,x = fy S (first yield)": "kN m",
    "M_y,z = fy S (first yield)": "kN m",
    "M_p,x = fy Z": "kN m",
    "M_p,z = fy Z": "kN m",
    "P_y = A fy": "kN",
}


def _positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s must be a positive finite number" % label) from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("%s must be a positive finite number" % label)
    return number


def _text(value):
    return str(value).strip() if value is not None else ""


class _Workbook:
    def __init__(self, archive):
        self.archive = archive
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationships = {rel.get("Id"): rel for rel in rels}
        self.sheets = {}
        for sheet in book.findall("s:sheets/s:sheet", _NS):
            name = sheet.get("name")
            if name in self.sheets:
                raise ValueError("Duplicate worksheet: %s" % name)
            relationship = relationships.get(sheet.get(_RID))
            if relationship is None or relationship.get("TargetMode") == "External":
                raise ValueError("Missing or external worksheet relationship: %s" % name)
            target = relationship.get("Target", "")
            self.sheets[name] = posixpath.normpath(
                target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
            )
        self.strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            self.strings = ["".join(node.text or "" for node in item.iter("{" + _NS["s"] + "}t"))
                            for item in root.findall("s:si", _NS)]

    def rows(self, name):
        if name not in self.sheets:
            raise ValueError("Missing worksheet: %s" % name)
        root = ET.fromstring(self.archive.read(self.sheets[name]))
        rows = []
        used_cells = set()
        for row in root.findall("s:sheetData/s:row", _NS):
            values = {}
            for cell in row.findall("s:c", _NS):
                address = cell.get("r", "")
                match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", address)
                if match is None or address in used_cells:
                    raise ValueError("Invalid or duplicate cell address in %s: %s" % (name, address))
                used_cells.add(address)
                column = match.group(1)
                kind = cell.get("t")
                if kind == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall("s:is//s:t", _NS))
                else:
                    cached = cell.find("s:v", _NS)
                    value = cached.text if cached is not None else None
                    if kind == "s" and value is not None:
                        try:
                            index = int(value)
                            if index < 0:
                                raise IndexError
                            value = self.strings[index]
                        except (ValueError, IndexError):
                            raise ValueError("Invalid shared string in %s: %s" % (name, address)) from None
                    elif kind in ("e", "b"):
                        value = None  # Spreadsheet errors and booleans are not section properties.
                values[column] = value
            rows.append(values)
        return rows


def _header(rows, names, sheet):
    candidates = []
    for index, row in enumerate(rows):
        labels = [_text(value) for value in row.values()]
        if all(name in labels for name in names):
            candidates.append((index, row, labels))
    if len(candidates) != 1:
        raise ValueError("Missing or duplicate required column headers in %s: %s" %
                         (sheet, ", ".join(names)))
    index, row, labels = candidates[0]
    if any(labels.count(name) != 1 for name in names):
        raise ValueError("Duplicate required column header in %s" % sheet)
    columns = {name: next(col for col, value in row.items() if _text(value) == name)
               for name in names}
    return index, columns


def _consistent_fy(actual, expected, label):
    actual = _positive(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Inconsistent report fy from %s: %.12g vs %.12g MPa" %
                         (label, actual, expected))


def _read_values(workbook):
    rows = workbook.rows(_SHEET)
    header, columns = _header(rows, ("property", "unit", _GROUP), _SHEET)
    values = {}
    seen = set()
    for row in rows[header + 1:]:
        prop = _text(row.get(columns["property"]))
        if not prop:
            continue
        if prop in seen:
            raise ValueError("Duplicate property in %s: %s" % (_SHEET, prop))
        seen.add(prop)
        if prop not in _PROPERTIES:
            continue
        unit = _text(row.get(columns["unit"]))
        if unit != _PROPERTIES[prop]:
            raise ValueError("Invalid unit for %s: expected %s, got %s" %
                             (prop, _PROPERTIES[prop], unit))
        values[prop] = _positive(row.get(columns[_GROUP]), _GROUP + " / " + prop)
    missing = set(_PROPERTIES) - set(values)
    if missing:
        raise ValueError("Missing built-up properties: %s" % ", ".join(sorted(missing)))
    return values


def _check_key_fy(workbook, report_fy):
    if "key_values" not in workbook.sheets:
        return
    rows = workbook.rows("key_values")
    header, columns = _header(rows, ("group", "item", "value", "unit"), "key_values")
    matches = [row for row in rows[header + 1:]
               if _text(row.get(columns["group"])) == "input"
               and _text(row.get(columns["item"])) == "fy"]
    if len(matches) > 1:
        raise ValueError("Duplicate input fy in key_values")
    if matches:
        row = matches[0]
        if _text(row.get(columns["unit"])) != "MPa":
            raise ValueError("Invalid unit for key_values input fy: expected MPa")
        _consistent_fy(row.get(columns["value"]), report_fy, "key_values input fy")


def read_reference(directory, fy=None):
    """Return JSON-safe gross built-up benchmarks from directory/summary.xlsx.

    fy is the current column yield stress in MPa, never an added-plate stress.
    Missing, ambiguous, nonphysical or inconsistent report data raises ValueError.
    The source workbook is read only and its SHA-256 records provenance.
    """
    path = (Path(directory).expanduser() / "summary.xlsx").resolve()
    raw = path.read_bytes()
    # Hash and parse the same bytes so provenance matches the parsed snapshot.
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            workbook = _Workbook(archive)
            values = _read_values(workbook)
            area = values["A"]
            report_fy = _positive(values["P_y = A fy"] * 1000.0 / area, "report fy")
            for axis in ("x", "z"):
                _consistent_fy(values["M_y,%s = fy S (first yield)" % axis] * 1e6 /
                               values["S_%s = I / c_max" % axis], report_fy, "My," + axis)
                _consistent_fy(values["M_p,%s = fy Z" % axis] * 1e6 /
                               values["Z_%s (plastic modulus)" % axis], report_fy, "Mp," + axis)
            _check_key_fy(workbook, report_fy)
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as error:
        raise ValueError("Invalid CUFSM workbook %s: %s" % (path, error)) from error
    current_fy = report_fy if fy is None else _positive(fy, "column fy")
    axes = {}
    for axis, angle in (("x", 0.0), ("z", 90.0)):
        elastic = values["S_%s = I / c_max" % axis]
        plastic = values["Z_%s (plastic modulus)" % axis]
        axes[axis] = {"axis_deg": angle, "S_mm3": elastic, "Z_mm3": plastic,
                      "SFy_kNm": _positive(elastic * current_fy / 1e6, "SFy," + axis),
                      "ZFy_kNm": _positive(plastic * current_fy / 1e6, "ZFy," + axis)}
    return {"schema": "cufsm_builtup_reference_v1", "source_workbook": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(), "group": _GROUP,
            "report_fy_MPa": report_fy, "fy_MPa": current_fy, "area_mm2": area,
            "AsFy_kN": _positive(area * current_fy / 1000.0, "AsFy"), "axes": axes}


def moment_reference(reference, angle_deg=0.0):
    """Integrate any model axis; keep the explicitly labelled legacy fallback.

    Legacy workbooks contain principal axes only. Other legacy angles return
    x benchmarks with matches_requested_axis=False, never a fabricated modulus.
    """
    try:
        angle = float(angle_deg)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Bending angle must be finite") from None
    if not math.isfinite(angle):
        raise ValueError("Bending angle must be finite")
    angle %= 180.0
    if reference.get('schema') in ('model_builtup_reference_v1','model_builtup_reference_v2'):
        import abaqus_load_case as loads
        data = loads.bending_reference(reference['section_segments'],reference['thickness_mm'],angle)
        elastic, plastic = data['section_modulus_outer_fibre_mm3'],data['plastic_modulus_mm3']
        fy = _positive(reference['fy_MPa'],'column fy')
        return dict(axis_deg=angle,S_mm3=elastic,Z_mm3=plastic,
                    SFy_kNm=elastic*fy/1e6,ZFy_kNm=plastic*fy/1e6,
                    axis_label='built-up %g deg' % angle,matches_requested_axis=True)
    if min(angle, 180.0 - angle) <= 1e-7:
        axis, label, matches = "x", "built-up x", True
    elif abs(angle - 90.0) <= 1e-7:
        axis, label, matches = "z", "built-up z", True
    else:
        axis, label, matches = "x", "report built-up x", False
    selected = dict(reference["axes"][axis])
    selected.update(axis_label=label, matches_requested_axis=matches)
    return selected


def read_family_geometry(directory):
    """Read the same four-piece CSV geometry as the native model builder."""
    import csv
    from abaqus_complete_model_m20 import read_geometry, validate_c4_geometry
    directory = Path(directory).expanduser().resolve()
    pieces, thickness, young, poisson = read_geometry(str(directory))
    validate_c4_geometry(pieces)
    with (directory/'builtup_member.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError('Expected exactly one member row: '+str(directory))
    member = rows[0]
    if not math.isclose(_positive(member.get('t_mm'), 'member thickness'), thickness,
                        rel_tol=0., abs_tol=1e-9):
        raise ValueError('Thickness differs between segment and member CSV files')
    length = _positive(member.get('L_mm'), 'member length')
    return dict(segments=[list(s) for k in sorted(pieces) for s in pieces[k]],
                thickness_mm=thickness, E_MPa=young, nu=poisson, length_mm=length,
                member=member)


def geometry_reference(segments, thickness, fy):
    """Gross FOUR-piece t*ds properties; no auxiliary transverse plates.

    S uses c_outer = c_centreline + t/2, matching the gross report benchmark.
    This plotted benchmark does not replace the shell load normalization.
    Z is integrated about the equal-area plastic neutral axis, not inferred
    from S or a symmetry multiplier. Actual shell thickness is used unchanged.
    """
    import json
    import abaqus_load_case as loads
    segments = [list(s) for s in loads._segments(segments)]
    thickness = _positive(thickness, 'shell thickness')
    fy = _positive(fy, 'column fy')
    props = loads.section_properties(segments, thickness)
    axes = {}
    for axis, angle in (('x',0.), ('z',90.)):
        data = loads.bending_reference(segments, thickness, angle)
        elastic, plastic = data['section_modulus_outer_fibre_mm3'], data['plastic_modulus_mm3']
        axes[axis] = dict(axis_deg=angle,S_mm3=elastic,Z_mm3=plastic,
                          SFy_kNm=elastic*fy/1e6,ZFy_kNm=plastic*fy/1e6)
    geometry = dict(section_segments=segments, thickness_mm=thickness)
    digest = hashlib.sha256(json.dumps(geometry,sort_keys=True,allow_nan=False).encode('ascii')).hexdigest()
    return dict(schema='model_builtup_reference_v2', source_sha256=digest,
                group=_GROUP, fy_MPa=fy, area_mm2=props['area_mm2'],
                AsFy_kN=props['area_mm2']*fy/1000., axes=axes,
                definition=props['definition'],
                elastic_modulus_basis='thin-wall I / (maximum centre-line distance + actual t/2)',
                **geometry)


def read_family_reference(directory, fy=None):
    data = read_family_geometry(directory)
    if fy is None:
        fy = data['member'].get('fy_MPa')
    return geometry_reference(data['segments'],data['thickness_mm'],fy)


def step5_reference(info):
    """Use the bound model snapshot, then family CSV, then embedded legacy data.

    Reading an old Excel path is never necessary. A changed bound model snapshot
    is an error, rather than silently substituting geometry for an existing run.
    HSS metadata has no built-up reference_pipeline and is left separate.
    """
    import json
    embedded = info.get('report_reference')
    fy = (info.get('settings') or {}).get('fy')
    if embedded and embedded.get('schema') in ('model_builtup_reference_v1','model_builtup_reference_v2'):
        if embedded['schema'] == 'model_builtup_reference_v2' and (fy is None or math.isclose(float(embedded['fy_MPa']),float(fy))):
            return embedded
        return geometry_reference(embedded['section_segments'],embedded['thickness_mm'],
                                  embedded['fy_MPa'] if fy is None else fy)
    reference = info.get('reference_pipeline') or {}
    run = reference.get('run_dir')
    if run:
        status = Path(run)/'pipeline_status.json'
        if status.is_file():
            raw = status.read_bytes()
            expected = reference.get('pipeline_status_sha256')
            if expected and hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Bound model snapshot changed: '+str(status))
            source = json.loads(raw.decode('utf-8-sig')).get('source_inputs') or {}
            if source.get('section_segments') and fy is not None:
                return geometry_reference(source['section_segments'],source['thickness_mm'],fy)
    source = reference.get('source_directory')
    if source and (Path(source)/'builtup_segments.csv').is_file() and fy is not None:
        return read_family_reference(source,fy)
    return embedded


def reference_origin(reference):
    return 'model geometry' if reference.get('schema') in ('model_builtup_reference_v1','model_builtup_reference_v2') else 'CUFSM report'
