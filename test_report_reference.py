"""Focused stdlib-only tests for the CUFSM built-up reference reader."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile


WORKSPACE = Path(__file__).resolve().parent
tempfile.tempdir = str(WORKSPACE)
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
PROPERTIES = [
    ("A", "mm^2", 4000.0),
    ("S_x = I / c_max", "mm^3", 500000.0),
    ("S_z = I / c_max", "mm^3", 600000.0),
    ("Z_x (plastic modulus)", "mm^3", 700000.0),
    ("Z_z (plastic modulus)", "mm^3", 800000.0),
    ("M_y,x = fy S (first yield)", "kN m", 120.0),
    ("M_y,z = fy S (first yield)", "kN m", 144.0),
    ("M_p,x = fy Z", "kN m", 168.0),
    ("M_p,z = fy Z", "kN m", 192.0),
    ("P_y = A fy", "kN", 960.0),
]


def _column(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def fixture(directory, properties=None, headers=None, key_fy=240.0,
            key_unit="MPa", sheet_name="section_properties_3", cached=True):
    """Write a tiny XLSX with mixed strings and misleading other groups."""
    headers = headers or ["group", "property", "unit", "single_piece",
                          "flat_single_piece", "builtup_4_pieces", "flat_builtup",
                          "box_same_A"]
    properties = PROPERTIES if properties is None else properties
    shared = []

    def sheet(rows):
        root = ET.Element("worksheet", xmlns=NS)
        data = ET.SubElement(root, "sheetData")
        for row_number, values in enumerate(rows, 1):
            row = ET.SubElement(data, "row", r=str(row_number))
            for col_number, value in enumerate(values, 1):
                if value is None:
                    continue
                cell = ET.SubElement(row, "c", r=_column(col_number) + str(row_number))
                if isinstance(value, str):
                    if (row_number + col_number) % 2:
                        cell.set("t", "inlineStr")
                        ET.SubElement(ET.SubElement(cell, "is"), "t").text = value
                    else:
                        cell.set("t", "s")
                        shared.append(value)
                        ET.SubElement(cell, "v").text = str(len(shared) - 1)
                else:
                    if row_number == 3:
                        ET.SubElement(cell, "f").text = "1+1"
                    if cached or row_number != 3:
                        ET.SubElement(cell, "v").text = str(value)
        return ET.tostring(root)

    rows = [["report title"], headers]
    for prop, unit, value in properties:
        mapping = {"group": "section", "property": prop, "unit": unit,
                   "builtup_4_pieces": value}
        rows.append([mapping.get(header, 1.0) for header in headers])
    worksheet = sheet(rows)
    keys = sheet([["group", "item", "value", "unit"],
                  ["input", "fy", key_fy, key_unit]]) if key_fy is not None else None
    book = ET.Element("workbook", xmlns=NS)
    sheets = ET.SubElement(book, "sheets")
    # The requested sheet deliberately follows a misleading unrelated sheet.
    ET.SubElement(sheets, "sheet", name="section_properties", sheetId="1",
                  attrib={"{" + REL + "}id": "rId1"})
    ET.SubElement(sheets, "sheet", name=sheet_name, sheetId="2",
                  attrib={"{" + REL + "}id": "rId2"})
    if keys is not None:
        ET.SubElement(sheets, "sheet", name="key_values", sheetId="3",
                      attrib={"{" + REL + "}id": "rId3"})
    relationships = ET.Element("Relationships", xmlns=PKG)
    for number in range(1, 4 if keys is not None else 3):
        ET.SubElement(relationships, "Relationship", Id="rId" + str(number),
                      Type=REL + "/worksheet",
                      Target="/xl/worksheets/part%d.xml" % number)
    strings = ET.Element("sst", xmlns=NS)
    for value in shared:
        # Rich text exercises concatenation, not just a single <t> lookup.
        item = ET.SubElement(strings, "si")
        for fragment in [value[:2], value[2:]]:
            ET.SubElement(ET.SubElement(item, "r"), "t").text = fragment
    path = Path(directory) / "summary.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", ET.tostring(book))
        archive.writestr("xl/_rels/workbook.xml.rels", ET.tostring(relationships))
        archive.writestr("xl/sharedStrings.xml", ET.tostring(strings))
        archive.writestr("xl/worksheets/part1.xml", worksheet)
        archive.writestr("xl/worksheets/part2.xml", worksheet)
        if keys is not None:
            archive.writestr("xl/worksheets/part3.xml", keys)
    return path


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(WORKSPACE))
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.reader = importlib.import_module("abaqus_report_reference")

    def test_reads_only_builtup_group_and_json_safe_provenance(self):
        path = fixture(self.directory)
        result = self.reader.read_reference(self.directory)
        self.assertEqual(result["schema"], "cufsm_builtup_reference_v1")
        self.assertEqual(result["group"], "builtup_4_pieces")
        self.assertEqual(result["source_workbook"], str(path.resolve()))
        self.assertEqual(result["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["area_mm2"], 4000.0)
        self.assertEqual(result["fy_MPa"], 240.0)
        self.assertEqual(result["report_fy_MPa"], 240.0)
        self.assertEqual(result["AsFy_kN"], 960.0)
        self.assertEqual(result["axes"]["x"], {"axis_deg": 0.0, "S_mm3": 500000.0,
                         "Z_mm3": 700000.0, "SFy_kNm": 120.0, "ZFy_kNm": 168.0})
        self.assertEqual(result["axes"]["z"]["SFy_kNm"], 144.0)
        self.assertEqual(result["axes"]["z"]["ZFy_kNm"], 192.0)
        json.dumps(result, allow_nan=False)

    def test_column_fy_override_recomputes_capacities(self):
        fixture(self.directory)
        result = self.reader.read_reference(self.directory, fy=300.0)
        self.assertEqual(result["fy_MPa"], 300.0)
        self.assertEqual(result["report_fy_MPa"], 240.0)
        self.assertEqual(result["AsFy_kN"], 1200.0)
        self.assertEqual(result["axes"]["x"]["SFy_kNm"], 150.0)
        self.assertEqual(result["axes"]["x"]["ZFy_kNm"], 210.0)

    def test_headers_sheets_and_properties_may_be_reordered(self):
        fixture(self.directory, list(reversed(PROPERTIES)),
                ["box_same_A", "unit", "builtup_4_pieces", "property", "group"], key_fy=None)
        result = self.reader.read_reference(self.directory)
        self.assertEqual(result["area_mm2"], 4000.0)
        self.assertEqual(result["axes"]["z"]["Z_mm3"], 800000.0)

    def test_missing_or_duplicate_headers_and_properties_fail(self):
        cases = [
            {"headers": ["group", "property", "unit", "single_piece"]},
            {"headers": ["property", "unit", "builtup_4_pieces", "builtup_4_pieces"]},
            {"headers": ["property", "property", "unit", "builtup_4_pieces"]},
            {"headers": ["property", "unit", "unit", "builtup_4_pieces"]},
            {"properties": PROPERTIES[:-1]},
            {"properties": PROPERTIES + [PROPERTIES[0]]},
            {"sheet_name": "other_section"},
        ]
        for case in cases:
            with self.subTest(case=case):
                fixture(self.directory, **case)
                with self.assertRaises(ValueError):
                    self.reader.read_reference(self.directory)

    def test_invalid_values_and_units_fail(self):
        for value in [0.0, -1.0, float("nan"), float("inf"), None, "bad"]:
            with self.subTest(value=value):
                properties = [("A", "mm^2", value)] + PROPERTIES[1:]
                fixture(self.directory, properties)
                with self.assertRaises(ValueError):
                    self.reader.read_reference(self.directory)
        fixture(self.directory, [("A", "cm^2", 4000.0)] + PROPERTIES[1:])
        with self.assertRaisesRegex(ValueError, "unit"):
            self.reader.read_reference(self.directory)

    def test_every_required_property_is_validated(self):
        for selected, _, _ in PROPERTIES:
            with self.subTest(property=selected):
                values = [(p, u, 0.0 if p == selected else v) for p, u, v in PROPERTIES]
                fixture(self.directory, values)
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    self.reader.read_reference(self.directory)

    def test_corrupt_workbook_is_rejected(self):
        (self.directory / "summary.xlsx").write_bytes(b"not an XLSX")
        with self.assertRaisesRegex(ValueError, "Invalid CUFSM workbook"):
            self.reader.read_reference(self.directory)

    def test_inconsistent_report_strengths_and_key_fy_fail(self):
        for prop in ["M_y,x = fy S (first yield)", "M_y,z = fy S (first yield)",
                     "M_p,x = fy Z", "M_p,z = fy Z", "P_y = A fy"]:
            with self.subTest(prop=prop):
                values = [(p, u, v * 1.05 if p == prop else v) for p, u, v in PROPERTIES]
                fixture(self.directory, values)
                with self.assertRaisesRegex(ValueError, "fy|consistent"):
                    self.reader.read_reference(self.directory, fy=300.0)
        for key_fy, unit in [(241.0, "MPa"), (240.0, "kPa")]:
            fixture(self.directory, key_fy=key_fy, key_unit=unit)
            with self.assertRaises(ValueError):
                self.reader.read_reference(self.directory)

    def test_uncached_formula_and_invalid_override_fail(self):
        fixture(self.directory, cached=False)
        with self.assertRaises(ValueError):
            self.reader.read_reference(self.directory)
        fixture(self.directory)
        for value in [0.0, -1.0, math.inf, math.nan, "bad"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.reader.read_reference(self.directory, fy=value)

    def test_axis_selection_and_diagonal_benchmark_are_explicit(self):
        fixture(self.directory)
        result = self.reader.read_reference(self.directory)
        for angle, axis in [(0.0, "x"), (180.0, "x"), (-180.0, "x"),
                            (90.0, "z"), (-90.0, "z"), (270.0, "z"),
                            (1e-9, "x"), (90.0 + 1e-9, "z")]:
            with self.subTest(angle=angle):
                selected = self.reader.moment_reference(result, angle)
                self.assertTrue(selected["matches_requested_axis"])
                self.assertEqual(selected["axis_label"], "built-up " + axis)
                self.assertEqual(selected["ZFy_kNm"], result["axes"][axis]["ZFy_kNm"])
        selected = self.reader.moment_reference(result, 45.0)
        self.assertFalse(selected["matches_requested_axis"])
        self.assertEqual(selected["axis_label"], "report built-up x")
        self.assertEqual(selected["axis_deg"], 0.0)
        self.assertEqual(selected["ZFy_kNm"], 168.0)
        self.assertNotIn("axis_label", result["axes"]["x"])
        for angle in [math.nan, math.inf, "bad"]:
            with self.assertRaises(ValueError):
                self.reader.moment_reference(result, angle)





class ModelReferenceTests(unittest.TestCase):
    def family(self, root):
        # Four rotations of one wall; exact thin-wall square: A=160, Ix=32000/3.
        rows = ['1,-10,10,10,10,2', '2,-10,-10,-10,10,2',
                '3,10,-10,-10,-10,2', '4,10,10,10,-10,2']
        (root/'builtup_segments.csv').write_text('\n'.join(rows)+'\n')
        (root/'builtup_member.csv').write_text(
            'L_mm,t_mm,E_MPa,nu,fy_MPa,A0_mm2,tag\n3600,2,200000,0.3,240,160,A160_t2_R2p5_lipS\n')
        return root

    def test_four_piece_properties_and_arbitrary_axis_need_no_workbook(self):
        import abaqus_report_reference as ref
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            family = self.family(Path(directory))
            result = ref.read_family_reference(family, fy=300.)
            self.assertAlmostEqual(result['area_mm2'], 160.)
            self.assertAlmostEqual(result['AsFy_kN'], 48.)
            self.assertAlmostEqual(result['axes']['x']['S_mm3'], (32000./3.)/11.)
            self.assertAlmostEqual(result['axes']['x']['Z_mm3'], 1200.)
            diagonal = ref.moment_reference(result,45.)
            self.assertTrue(diagonal['matches_requested_axis'])
            self.assertAlmostEqual(diagonal['Z_mm3'],800.*math.sqrt(2))
            self.assertAlmostEqual(diagonal['S_mm3'], (32000./3.)/(10.*math.sqrt(2)+1.))
            self.assertNotIn('source_workbook',result)
            json.dumps(result,allow_nan=False)

    def test_old_step5_metadata_can_use_geometry_without_reading_its_old_report(self):
        import abaqus_step5_force_displacement_core as fd
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            family = self.family(Path(directory))
            info = dict(settings=dict(fy=240.), reference_pipeline=dict(source_directory=str(family)),
                        report_reference=dict(source_workbook='deleted.xlsx'))
            current = fd.with_report_reference(info,None)
            self.assertAlmostEqual(current['report_reference']['area_mm2'],160.)
            self.assertEqual(info['report_reference'],dict(source_workbook='deleted.xlsx'))

    def test_csv_thickness_conflict_fails_before_build(self):
        import abaqus_report_reference as ref
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            family=self.family(Path(directory))
            path=family/'builtup_member.csv'
            path.write_text(path.read_text().replace('3600,2,','3600,3,'))
            with self.assertRaisesRegex(ValueError,'[Tt]hickness'):
                ref.read_family_reference(family,fy=240.)

    def test_existing_model_snapshot_wins_over_changed_source_csv_and_checks_hash(self):
        import abaqus_report_reference as ref
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            root=Path(directory); family=self.family(root)
            source=ref.read_family_geometry(family)
            status=root/'pipeline_status.json'
            status.write_text(json.dumps(dict(source_inputs=dict(section_segments=source['segments'],thickness_mm=2.))))
            info=dict(settings=dict(fy=240.),reference_pipeline=dict(run_dir=str(root),
                source_directory=str(family),pipeline_status_sha256=hashlib.sha256(status.read_bytes()).hexdigest()))
            (root/'builtup_segments.csv').write_text('invalid changed CSV')
            self.assertAlmostEqual(ref.step5_reference(info)['area_mm2'],160.)
            status.write_text(status.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):
                ref.step5_reference(info)

    def test_family_references_do_not_leak_between_members(self):
        from types import SimpleNamespace
        import abaqus_family_pipeline as pipeline
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            root=Path(directory); a=root/'a'; b=root/'b'; a.mkdir(); b.mkdir()
            self.family(a); self.family(b)
            path=b/'builtup_segments.csv'; path.write_text(path.read_text().replace(',2\n',',3\n'))
            path=b/'builtup_member.csv'; path.write_text(path.read_text().replace('3600,2,','3600,3,'))
            args=SimpleNamespace(cufsm_report_dir=None,report_reference=None,steel_column_plate_fy=240.)
            first=pipeline.family_reference_args(args,a)
            second=pipeline.family_reference_args(first,b)
            self.assertAlmostEqual(first.report_reference['area_mm2'],160.)
            self.assertAlmostEqual(second.report_reference['area_mm2'],240.)
            self.assertIsNone(args.report_reference)
    def test_old_centreline_reference_migrates_in_memory_without_changing_loads(self):
        import copy
        import abaqus_report_reference as ref
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            family=self.family(Path(directory))
            current=ref.read_family_reference(family,240.)
            old=copy.deepcopy(current); old['schema']='model_builtup_reference_v1'
            old['axes']['x']['S_mm3']=3200./3.
            original=copy.deepcopy(old)
            info=dict(settings=dict(fy=240.),report_reference=old,reference_moment_Nmm=123456.)
            selected=ref.step5_reference(info)
            self.assertEqual(selected['schema'],'model_builtup_reference_v2')
            self.assertAlmostEqual(selected['axes']['x']['S_mm3'],(32000./3.)/11.)
            self.assertEqual(old,original)
            self.assertEqual(info['reference_moment_Nmm'],123456.)



if __name__ == "__main__":
    unittest.main()
