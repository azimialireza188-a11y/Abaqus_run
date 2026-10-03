# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-16.25.25 RELr426 190762
# Run by 810200014 on Sat Oct  3 13:25:24 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.957031, 0.954861), 
    width=140.875, height=94.7222)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('abaqus_complete_model_m20.py', __main__.__dict__)
#: [13:25:24] Output directory: D:\CFS-Column\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15
#: [13:25:24] Settings: {"build_only": false, "builtup_dir": "C:\\Users\\810200014.HAMI.000\\Documents\\CUFSM-Single\\cfs_abaqus\\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15", "check_inputs": false, "cpus": 8, "max_iterations": 1250, "mesh_mm": 20.0, "n_modes": 250, "n_vectors": 500, "output_dir": "D:\\CFS-Column\\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15", "output_root": null, "resume_post": null, "skip_post": false}
#: [13:25:24] 1/4 BUILD: geometry, mesh, contact, CAE and INP.
#: Source inputs: {"E_MPa": 200000.0, "bolt_positions_mm": [25.0, 222.2222222, 419.4444444, 616.6666667, 813.8888889, 1011.111111, 1208.333333, 1405.555556, 1602.777778, 1800.0, 1997.222222, 2194.444444, 2391.666667, 2588.888889, 2786.111111, 2983.333333, 3180.555556, 3377.777778, 3575.0], "bolt_row_mm": 15.0, "bolt_solids_holes_contact_pretension": false, "bolts_per_seam": 19, "clear_gap_mm": 10.0, "connection_model": "BEAM_MPC", "expected_links": 76, "finish_end_mm": 25.0, "length_mm": 3600.0, "nu": 0.3, "pitch_mm": [197.2222222, 197.2222222, 197.2222223, 197.2222222, 197.2222221, 197.222222, 197.222223, 197.222222, 197.222222, 197.222222, 197.222222, 197.222223, 197.222222, 197.222222, 197.222222, 197.222223, 197.222222, 197.222222], "shell_contact": "GENERAL_STANDARD_HARD_FRICTIONLESS", "source_directory": "C:\\Users\\810200014.HAMI.000\\Documents\\CUFSM-Single\\cfs_abaqus\\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15", "start_end_mm": 25.0, "thickness_mm": 2.0}
#: The model "BU_BOLT_L3600_M20" has been created.
#: Partitions ready; simplifying only redundant section boundaries.
#: Before meshing: 100 faces; edge counts [4]
#: After meshing: 4255 nodes, 4048 elements; unmeshed=None
#: Mesh: 100 faces, 4255 nodes, 4048 S4R per piece.
#: The interaction property "Hard_Frictionless" has been created.
#: The interaction "GeneralContact" has been created.
#: The model database has been saved to "D:\CFS-Column\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15\BU_BOLT_L3600_M20.cae".
#: BUILD COMPLETE: CAE and INP saved. No analysis submitted yet. {'length_mm': 3600.0, 'target_mesh_mm': 20.0, 'modes': 250, 'vectors': 500, 'max_iterations': 1250, 'cpus': 8, 'job_name': 'BU_BOLT_L3600_M20', 'odb': 'D:\\CFS-Column\\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15\\BU_BOLT_L3600_M20.odb', 'nodes': 17020, 'elements': 16192, 'element_type': 'S4R', 'bolts_per_seam': 19, 'rigid_links': 76, 'boundary_conditions': 9, 'loads': 8, 'reference_stress_MPa': 1.0, 'submitted': False, 'contact': {'type': 'General contact (Standard)', 'step': 'Initial', 'domain': 'All exterior surfaces, including self-contact', 'normal': 'HARD', 'tangential': 'FRICTIONLESS', 'allow_separation': True, 'buckle_limitation': 'Contact status fixed at the base state'}, 'source_inputs': {'source_directory': 'C:\\Users\\810200014.HAMI.000\\Documents\\CUFSM-Single\\cfs_abaqus\\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15', 'length_mm': 3600.0, 'thickness_mm': 2.0, 'E_MPa': 200000.0, 'nu': 0.3, 'clear_gap_mm': 10.0, 'bolt_row_mm': 15.0, 'bolts_per_seam': 19, 'start_end_mm': 25.0, 'finish_end_mm': 25.0, 'pitch_mm': [197.2222222, 197.2222222, 197.2222223, 197.2222222, 197.2222221, 197.222222, 197.222223, 197.222222, 197.222222, 197.222222, 197.222222, 197.222223, 197.222222, 197.222222, 197.222222, 197.222223, 197.222222, 197.222222], 'bolt_positions_mm': [25.0, 222.2222222, 419.4444444, 616.6666667, 813.8888889, 1011.111111, 1208.333333, 1405.555556, 1602.777778, 1800.0, 1997.222222, 2194.444444, 2391.666667, 2588.888889, 2786.111111, 2983.333333, 3180.555556, 3377.777778, 3575.0], 'expected_links': 76, 'connection_model': 'BEAM_MPC', 'bolt_solids_holes_contact_pretension': False, 'shell_contact': 'GENERAL_STANDARD_HARD_FRICTIONLESS'}}
#: [13:25:59] 2/4 SOLVE: submitting BU_BOLT_L3600_M20
#: [13:50:00] Solver completed successfully: Successful completion in .sta, .log; ODB present and unlocked
#: [13:50:00] 3/4 POSTPROCESS: all modes from D:\CFS-Column\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15\BU_BOLT_L3600_M20.odb
#: Model: D:/CFS-Column/A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15/BU_BOLT_L3600_M20.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     4
#: Number of Meshes:             4
#: Number of Element Sets:       9
#: Number of Node Sets:          162
#: Number of Steps:              1
#: Mode 1: eigenvalue=390.44, n=13, half-wave=276.923076923077 mm, share=0.992, dominant
#: Mode 2: eigenvalue=391.54, n=14, half-wave=257.142857142857 mm, share=0.934, dominant
#: Mode 3: eigenvalue=393.37, n=12, half-wave=300.0 mm, share=0.962, dominant
#: Mode 4: eigenvalue=395.66, n=15, half-wave=240.0 mm, share=0.987, dominant
#: Mode 5: eigenvalue=399.97, n=13, half-wave=276.923076923077 mm, share=0.984, dominant
#: Mode 6: eigenvalue=399.97, n=13, half-wave=276.923076923077 mm, share=0.984, dominant
#: Mode 7: eigenvalue=400.9, n=12, half-wave=300.0 mm, share=0.527, mixed
#: Mode 8: eigenvalue=400.9, n=12, half-wave=300.0 mm, share=0.527, mixed
#: Mode 9: eigenvalue=401.67, n=11, half-wave=327.272727272727 mm, share=0.996, dominant
#: Mode 10: eigenvalue=401.69, n=12, half-wave=300.0 mm, share=0.622, mixed
#: Mode 11: eigenvalue=401.69, n=12, half-wave=300.0 mm, share=0.622, mixed
#: Mode 12: eigenvalue=401.76, n=16, half-wave=225.0 mm, share=0.993, dominant
#: Mode 13: eigenvalue=404.32, n=15, half-wave=240.0 mm, share=0.955, dominant
#: Mode 14: eigenvalue=404.32, n=15, half-wave=240.0 mm, share=0.955, dominant
#: Mode 15: eigenvalue=406.36, n=11, half-wave=327.272727272727 mm, share=0.984, dominant
#: Mode 16: eigenvalue=406.36, n=11, half-wave=327.272727272727 mm, share=0.984, dominant
#: Mode 17: eigenvalue=408.7, n=16, half-wave=225.0 mm, share=0.961, dominant
#: Mode 18: eigenvalue=408.7, n=16, half-wave=225.0 mm, share=0.961, dominant
#: Mode 19: eigenvalue=409.35, n=17, half-wave=211.764705882353 mm, share=0.983, dominant
#: Mode 20: eigenvalue=410.28, n=12, half-wave=300.0 mm, share=0.969, dominant
#: Mode 21: eigenvalue=410.73, n=13, half-wave=276.923076923077 mm, share=0.657, mixed
#: Mode 22: eigenvalue=411.73, n=11, half-wave=327.272727272727 mm, share=0.783, dominant
#: Mode 23: eigenvalue=412.81, n=14, half-wave=257.142857142857 mm, share=0.936, dominant
#: Mode 24: eigenvalue=413.38, n=18, half-wave=200.0 mm, share=0.855, dominant
#: Mode 25: eigenvalue=413.46, n=17, half-wave=211.764705882353 mm, share=0.881, dominant
#: Mode 26: eigenvalue=413.46, n=17, half-wave=211.764705882353 mm, share=0.881, dominant
#: Mode 27: eigenvalue=415, n=18, half-wave=200.0 mm, share=0.921, dominant
#: Mode 28: eigenvalue=415, n=18, half-wave=200.0 mm, share=0.921, dominant
#: Mode 29: eigenvalue=415.32, n=15, half-wave=240.0 mm, share=0.948, dominant
#: Mode 30: eigenvalue=415.84, n=10, half-wave=360.0 mm, share=0.982, dominant
#: Mode 31: eigenvalue=416.63, n=18, half-wave=200.0 mm, share=0.946, dominant
#: Mode 32: eigenvalue=416.81, n=10, half-wave=360.0 mm, share=0.997, dominant
#: Mode 33: eigenvalue=416.81, n=10, half-wave=360.0 mm, share=0.997, dominant
#: Mode 34: eigenvalue=417.42, n=17, half-wave=211.764705882353 mm, share=0.580, mixed
#: Mode 35: eigenvalue=417.92, n=10, half-wave=360.0 mm, share=0.994, dominant
#: Mode 36: eigenvalue=418.65, n=16, half-wave=225.0 mm, share=0.923, dominant
#: Mode 37: eigenvalue=424.19, n=17, half-wave=211.764705882353 mm, share=0.807, dominant
#: Mode 38: eigenvalue=424.25, n=18, half-wave=200.0 mm, share=0.851, dominant
#: Mode 39: eigenvalue=425.23, n=16, half-wave=225.0 mm, share=0.609, mixed
#: Mode 40: eigenvalue=425.56, n=9, half-wave=400.0 mm, share=0.982, dominant
#: Mode 41: eigenvalue=425.94, n=17, half-wave=211.764705882353 mm, share=0.670, mixed
#: Mode 42: eigenvalue=425.94, n=17, half-wave=211.764705882353 mm, share=0.670, mixed
#: Mode 43: eigenvalue=426.05, n=18, half-wave=200.0 mm, share=0.855, dominant
#: Mode 44: eigenvalue=426.05, n=18, half-wave=200.0 mm, share=0.855, dominant
#: Mode 45: eigenvalue=428.22, n=18, half-wave=200.0 mm, share=0.983, dominant
#: Mode 46: eigenvalue=428.53, n=17, half-wave=211.764705882353 mm, share=0.556, mixed
#: Mode 47: eigenvalue=431.3, n=19, half-wave=189.473684210526 mm, share=0.978, dominant
#: Mode 48: eigenvalue=431.63, n=16, half-wave=225.0 mm, share=0.531, mixed
#: Mode 49: eigenvalue=431.63, n=16, half-wave=225.0 mm, share=0.531, mixed
#: Mode 50: eigenvalue=434.53, n=15, half-wave=240.0 mm, share=0.568, mixed
#: Mode 51: eigenvalue=435.16, n=16, half-wave=225.0 mm, share=0.515, mixed
#: Mode 52: eigenvalue=435.55, n=9, half-wave=400.0 mm, share=0.951, dominant
#: Mode 53: eigenvalue=435.55, n=9, half-wave=400.0 mm, share=0.951, dominant
#: Mode 54: eigenvalue=436.17, n=19, half-wave=189.473684210526 mm, share=0.478, mixed
#: Mode 55: eigenvalue=436.17, n=19, half-wave=189.473684210526 mm, share=0.478, mixed
#: Mode 56: eigenvalue=440.49, n=19, half-wave=189.473684210526 mm, share=0.962, dominant
#: Mode 57: eigenvalue=440.54, n=15, half-wave=240.0 mm, share=0.528, mixed
#: Mode 58: eigenvalue=440.54, n=15, half-wave=240.0 mm, share=0.528, mixed
#: Mode 59: eigenvalue=442.63, n=20, half-wave=180.0 mm, share=0.975, dominant
#: Mode 60: eigenvalue=444.27, n=15, half-wave=240.0 mm, share=0.508, mixed
#: Mode 61: eigenvalue=444.52, n=8, half-wave=450.0 mm, share=0.997, dominant
#: Mode 62: eigenvalue=445.6, n=9, half-wave=400.0 mm, share=0.998, dominant
#: Mode 63: eigenvalue=446.03, n=20, half-wave=180.0 mm, share=0.959, dominant
#: Mode 64: eigenvalue=446.03, n=20, half-wave=180.0 mm, share=0.959, dominant
#: Mode 65: eigenvalue=447.24, n=14, half-wave=257.142857142857 mm, share=0.550, mixed
#: Mode 66: eigenvalue=449.33, n=20, half-wave=180.0 mm, share=0.906, dominant
#: Mode 67: eigenvalue=452.44, n=14, half-wave=257.142857142857 mm, share=0.537, mixed
#: Mode 68: eigenvalue=452.44, n=14, half-wave=257.142857142857 mm, share=0.537, mixed
#: Mode 69: eigenvalue=455.47, n=21, half-wave=171.428571428571 mm, share=0.954, dominant
#: Mode 70: eigenvalue=455.93, n=14, half-wave=257.142857142857 mm, share=0.507, mixed
#: Mode 71: eigenvalue=457.69, n=21, half-wave=171.428571428571 mm, share=0.951, dominant
#: Mode 72: eigenvalue=457.69, n=21, half-wave=171.428571428571 mm, share=0.951, dominant
#: Mode 73: eigenvalue=459.75, n=21, half-wave=171.428571428571 mm, share=0.935, dominant
#: Mode 74: eigenvalue=463.26, n=13, half-wave=276.923076923077 mm, share=0.546, mixed
#: Mode 75: eigenvalue=467.12, n=13, half-wave=276.923076923077 mm, share=0.528, mixed
#: Mode 76: eigenvalue=467.12, n=13, half-wave=276.923076923077 mm, share=0.528, mixed
#: Mode 77: eigenvalue=467.79, n=8, half-wave=450.0 mm, share=0.996, dominant
#: Mode 78: eigenvalue=467.79, n=8, half-wave=450.0 mm, share=0.996, dominant
#: Mode 79: eigenvalue=469.6, n=22, half-wave=163.636363636364 mm, share=0.944, dominant
#: Mode 80: eigenvalue=469.97, n=13, half-wave=276.923076923077 mm, share=0.496, mixed
#: Mode 81: eigenvalue=470.94, n=22, half-wave=163.636363636364 mm, share=0.821, dominant
#: Mode 82: eigenvalue=470.94, n=22, half-wave=163.636363636364 mm, share=0.821, dominant
#: Mode 83: eigenvalue=472.1, n=22, half-wave=163.636363636364 mm, share=0.959, dominant
#: Mode 84: eigenvalue=480.01, n=7, half-wave=514.285714285714 mm, share=0.998, dominant
#: Mode 85: eigenvalue=481.73, n=12, half-wave=300.0 mm, share=0.530, mixed
#: Mode 86: eigenvalue=484.15, n=12, half-wave=300.0 mm, share=0.503, mixed
#: Mode 87: eigenvalue=484.15, n=12, half-wave=300.0 mm, share=0.503, mixed
#: Mode 88: eigenvalue=485.04, n=23, half-wave=156.521739130435 mm, share=0.907, dominant
#: Mode 89: eigenvalue=485.49, n=23, half-wave=156.521739130435 mm, share=0.910, dominant
#: Mode 90: eigenvalue=485.49, n=23, half-wave=156.521739130435 mm, share=0.910, dominant
#: Mode 91: eigenvalue=486.03, n=23, half-wave=156.521739130435 mm, share=0.843, dominant
#: Mode 92: eigenvalue=486.05, n=12, half-wave=300.0 mm, share=0.472, mixed
#: Mode 93: eigenvalue=490.73, n=8, half-wave=450.0 mm, share=0.999, dominant
#: Mode 94: eigenvalue=501.5, n=24, half-wave=150.0 mm, share=0.896, dominant
#: Mode 95: eigenvalue=501.65, n=24, half-wave=150.0 mm, share=0.883, dominant
#: Mode 96: eigenvalue=501.65, n=24, half-wave=150.0 mm, share=0.883, dominant
#: Mode 97: eigenvalue=501.81, n=24, half-wave=150.0 mm, share=0.855, dominant
#: Mode 98: eigenvalue=501.87, n=11, half-wave=327.272727272727 mm, share=0.476, mixed
#: Mode 99: eigenvalue=502.86, n=11, half-wave=327.272727272727 mm, share=0.454, mixed
#: Mode 100: eigenvalue=502.86, n=11, half-wave=327.272727272727 mm, share=0.454, mixed
#: Mode 101: eigenvalue=503.75, n=11, half-wave=327.272727272727 mm, share=0.415, mixed
#: Mode 102: eigenvalue=514.5, n=1, half-wave=3600.0 mm, share=1.000, dominant
#: Mode 103: eigenvalue=518.9, n=25, half-wave=144.0 mm, share=0.800, dominant
#: Mode 104: eigenvalue=519.26, n=25, half-wave=144.0 mm, share=0.508, mixed
#: Mode 105: eigenvalue=519.26, n=25, half-wave=144.0 mm, share=0.508, mixed
#: Mode 106: eigenvalue=519.9, n=25, half-wave=144.0 mm, share=0.795, dominant
#: Mode 107: eigenvalue=522.29, n=7, half-wave=514.285714285714 mm, share=0.990, dominant
#: Mode 108: eigenvalue=522.29, n=7, half-wave=514.285714285714 mm, share=0.990, dominant
#: Mode 109: eigenvalue=522.31, n=26, half-wave=138.461538461538 mm, share=0.513, mixed
#: Mode 110: eigenvalue=522.39, n=26, half-wave=138.461538461538 mm, share=0.502, mixed
#: Mode 111: eigenvalue=522.39, n=26, half-wave=138.461538461538 mm, share=0.502, mixed
#: Mode 112: eigenvalue=522.52, n=26, half-wave=138.461538461538 mm, share=0.491, mixed
#: Mode 113: eigenvalue=538.47, n=26, half-wave=138.461538461538 mm, share=0.579, mixed
#: Mode 114: eigenvalue=538.99, n=26, half-wave=138.461538461538 mm, share=0.717, dominant
#: Mode 115: eigenvalue=538.99, n=26, half-wave=138.461538461538 mm, share=0.717, dominant
#: Mode 116: eigenvalue=539.45, n=26, half-wave=138.461538461538 mm, share=0.745, dominant
#: Mode 117: eigenvalue=541.84, n=27, half-wave=133.333333333333 mm, share=0.547, mixed
#: Mode 118: eigenvalue=542.06, n=27, half-wave=133.333333333333 mm, share=0.586, mixed
#: Mode 119: eigenvalue=542.06, n=27, half-wave=133.333333333333 mm, share=0.586, mixed
#: Mode 120: eigenvalue=542.2, n=27, half-wave=133.333333333333 mm, share=0.630, mixed
#: Mode 121: eigenvalue=546.15, n=6, half-wave=600.0 mm, share=0.999, dominant
#: Mode 122: eigenvalue=560.07, n=27, half-wave=133.333333333333 mm, share=0.456, mixed
#: Mode 123: eigenvalue=560.22, n=27, half-wave=133.333333333333 mm, share=0.628, mixed
#: Mode 124: eigenvalue=560.22, n=27, half-wave=133.333333333333 mm, share=0.628, mixed
#: Mode 125: eigenvalue=560.26, n=27, half-wave=133.333333333333 mm, share=0.611, mixed
#: Mode 126: eigenvalue=561.43, n=28, half-wave=128.571428571429 mm, share=0.633, mixed
#: Mode 127: eigenvalue=561.67, n=28, half-wave=128.571428571429 mm, share=0.670, mixed
#: Mode 128: eigenvalue=561.67, n=28, half-wave=128.571428571429 mm, share=0.670, mixed
#: Mode 129: eigenvalue=561.7, n=28, half-wave=128.571428571429 mm, share=0.691, mixed
#: Mode 130: eigenvalue=564.15, n=7, half-wave=514.285714285714 mm, share=0.993, dominant
#: Mode 131: eigenvalue=581.06, n=29, half-wave=124.137931034483 mm, share=0.763, dominant
#: Mode 132: eigenvalue=581.34, n=29, half-wave=124.137931034483 mm, share=0.570, mixed
#: Mode 133: eigenvalue=581.42, n=29, half-wave=124.137931034483 mm, share=0.671, mixed
#: Mode 134: eigenvalue=581.42, n=29, half-wave=124.137931034483 mm, share=0.671, mixed
#: Mode 135: eigenvalue=581.96, n=28, half-wave=128.571428571429 mm, share=0.695, mixed
#: Mode 136: eigenvalue=582.46, n=28, half-wave=128.571428571429 mm, share=0.628, mixed
#: Mode 137: eigenvalue=582.46, n=28, half-wave=128.571428571429 mm, share=0.628, mixed
#: Mode 138: eigenvalue=583.07, n=28, half-wave=128.571428571429 mm, share=0.599, mixed
#: Mode 139: eigenvalue=598.9, n=6, half-wave=600.0 mm, share=0.739, dominant
#: Mode 140: eigenvalue=598.9, n=6, half-wave=600.0 mm, share=0.739, dominant
#: Mode 141: eigenvalue=600.72, n=30, half-wave=120.0 mm, share=0.755, dominant
#: Mode 142: eigenvalue=600.78, n=30, half-wave=120.0 mm, share=0.729, dominant
#: Mode 143: eigenvalue=604.12, n=29, half-wave=124.137931034483 mm, share=0.682, mixed
#: Mode 144: eigenvalue=605.24, n=29, half-wave=124.137931034483 mm, share=0.589, mixed
#: Mode 145: eigenvalue=605.24, n=29, half-wave=124.137931034483 mm, share=0.589, mixed
#: Mode 146: eigenvalue=605.88, n=29, half-wave=124.137931034483 mm, share=0.622, mixed
#: Mode 147: eigenvalue=614.67, n=6, half-wave=600.0 mm, share=0.978, dominant
#: Mode 148: eigenvalue=614.67, n=6, half-wave=600.0 mm, share=0.978, dominant
#: Mode 149: eigenvalue=619.78, n=31, half-wave=116.129032258065 mm, share=0.698, mixed
#: Mode 150: eigenvalue=620.18, n=31, half-wave=116.129032258065 mm, share=0.719, dominant
#: Mode 151: eigenvalue=620.18, n=31, half-wave=116.129032258065 mm, share=0.719, dominant
#: Mode 152: eigenvalue=620.67, n=31, half-wave=116.129032258065 mm, share=0.742, dominant
#: Mode 153: eigenvalue=626.17, n=30, half-wave=120.0 mm, share=0.733, dominant
#: Mode 154: eigenvalue=628.53, n=30, half-wave=120.0 mm, share=0.680, mixed
#: Mode 155: eigenvalue=631.5, n=6, half-wave=600.0 mm, share=0.883, dominant
#: Mode 156: eigenvalue=631.5, n=6, half-wave=600.0 mm, share=0.883, dominant
#: Mode 157: eigenvalue=639.66, n=32, half-wave=112.5 mm, share=0.859, dominant
#: Mode 158: eigenvalue=640.11, n=32, half-wave=112.5 mm, share=0.798, dominant
#: Mode 159: eigenvalue=640.11, n=32, half-wave=112.5 mm, share=0.798, dominant
#: Mode 160: eigenvalue=640.5, n=32, half-wave=112.5 mm, share=0.707, dominant
#: Mode 161: eigenvalue=648.18, n=31, half-wave=116.129032258065 mm, share=0.733, dominant
#: Mode 162: eigenvalue=648.66, n=31, half-wave=116.129032258065 mm, share=0.587, mixed
#: Mode 163: eigenvalue=648.66, n=31, half-wave=116.129032258065 mm, share=0.587, mixed
#: Mode 164: eigenvalue=650.32, n=31, half-wave=116.129032258065 mm, share=0.577, mixed
#: Mode 165: eigenvalue=659.08, n=33, half-wave=109.090909090909 mm, share=0.706, dominant
#: Mode 166: eigenvalue=659.66, n=33, half-wave=109.090909090909 mm, share=0.690, mixed
#: Mode 167: eigenvalue=659.66, n=33, half-wave=109.090909090909 mm, share=0.690, mixed
#: Mode 168: eigenvalue=660.28, n=33, half-wave=109.090909090909 mm, share=0.717, dominant
#: Mode 169: eigenvalue=669.82, n=2, half-wave=1800.0 mm, share=0.715, dominant
#: Mode 170: eigenvalue=670.73, n=32, half-wave=112.5 mm, share=0.785, dominant
#: Mode 171: eigenvalue=670.73, n=32, half-wave=112.5 mm, share=0.785, dominant
#: Mode 172: eigenvalue=671.58, n=2, half-wave=1800.0 mm, share=0.994, dominant
#: Mode 173: eigenvalue=671.91, n=32, half-wave=112.5 mm, share=0.809, dominant
#: Mode 174: eigenvalue=672.86, n=5, half-wave=720.0 mm, share=0.999, dominant
#: Mode 175: eigenvalue=678.48, n=34, half-wave=105.882352941176 mm, share=0.735, dominant
#: Mode 176: eigenvalue=678.66, n=34, half-wave=105.882352941176 mm, share=0.513, mixed
#: Mode 177: eigenvalue=678.66, n=34, half-wave=105.882352941176 mm, share=0.513, mixed
#: Mode 178: eigenvalue=687.07, n=6, half-wave=600.0 mm, share=0.995, dominant
#: Mode 179: eigenvalue=689.25, n=2, half-wave=1800.0 mm, share=0.995, dominant
#: Mode 180: eigenvalue=690.98, n=33, half-wave=109.090909090909 mm, share=0.647, mixed
#: Mode 181: eigenvalue=691.58, n=33, half-wave=109.090909090909 mm, share=0.444, mixed
#: Mode 182: eigenvalue=691.58, n=33, half-wave=109.090909090909 mm, share=0.444, mixed
#: Mode 183: eigenvalue=692.53, n=33, half-wave=109.090909090909 mm, share=0.727, dominant
#: Mode 184: eigenvalue=697.29, n=1, half-wave=3600.0 mm, share=0.875, dominant
#: Mode 185: eigenvalue=697.29, n=1, half-wave=3600.0 mm, share=0.875, dominant
#: Mode 186: eigenvalue=697.84, n=35, half-wave=102.857142857143 mm, share=0.604, mixed
#: Mode 187: eigenvalue=700.89, n=1, half-wave=3600.0 mm, share=0.912, dominant
#: Mode 188: eigenvalue=707.28, n=19, half-wave=189.473684210526 mm, share=0.522, mixed
#: Mode 189: eigenvalue=710.11, n=19, half-wave=189.473684210526 mm, share=0.534, mixed
#: Mode 190: eigenvalue=710.11, n=19, half-wave=189.473684210526 mm, share=0.534, mixed
#: Mode 191: eigenvalue=711.73, n=2, half-wave=1800.0 mm, share=0.535, mixed
#: Mode 192: eigenvalue=711.73, n=2, half-wave=1800.0 mm, share=0.535, mixed
#: Mode 193: eigenvalue=712.79, n=34, half-wave=105.882352941176 mm, share=0.577, mixed
#: Mode 194: eigenvalue=713.8, n=2, half-wave=1800.0 mm, share=0.886, dominant
#: Mode 195: eigenvalue=714.06, n=19, half-wave=189.473684210526 mm, share=0.534, mixed
#: Mode 196: eigenvalue=716.59, n=2, half-wave=1800.0 mm, share=0.720, dominant
#: Mode 197: eigenvalue=716.96, n=36, half-wave=100.0 mm, share=0.430, mixed
#: Mode 198: eigenvalue=716.96, n=36, half-wave=100.0 mm, share=0.430, mixed
#: Mode 199: eigenvalue=718.04, n=36, half-wave=100.0 mm, share=0.820, dominant
#: Mode 200: eigenvalue=718.48, n=2, half-wave=1800.0 mm, share=0.820, dominant
#: Mode 201: eigenvalue=718.94, n=16, half-wave=225.0 mm, share=0.536, mixed
#: Mode 202: eigenvalue=718.94, n=16, half-wave=225.0 mm, share=0.536, mixed
#: Mode 203: eigenvalue=721.78, n=16, half-wave=225.0 mm, share=0.580, mixed
#: Mode 204: eigenvalue=729.02, n=15, half-wave=240.0 mm, share=0.524, mixed
#: Mode 205: eigenvalue=730.19, n=1, half-wave=3600.0 mm, share=0.450, mixed
#: Mode 206: eigenvalue=730.19, n=1, half-wave=3600.0 mm, share=0.450, mixed
#: Mode 207: eigenvalue=732.07, n=15, half-wave=240.0 mm, share=0.470, mixed
#: Mode 208: eigenvalue=732.55, n=1, half-wave=3600.0 mm, share=0.822, dominant
#: Mode 209: eigenvalue=732.55, n=1, half-wave=3600.0 mm, share=0.822, dominant
#: Mode 210: eigenvalue=733.13, n=1, half-wave=3600.0 mm, share=0.772, dominant
#: Mode 211: eigenvalue=734.55, n=15, half-wave=240.0 mm, share=0.309, mixed
#: Mode 212: eigenvalue=736.56, n=1, half-wave=3600.0 mm, share=0.739, dominant
#: Mode 213: eigenvalue=736.56, n=1, half-wave=3600.0 mm, share=0.739, dominant
#: Mode 214: eigenvalue=736.8, n=37, half-wave=97.2972972972973 mm, share=0.516, mixed
#: Mode 215: eigenvalue=742.98, n=1, half-wave=3600.0 mm, share=0.961, dominant
#: Mode 216: eigenvalue=745.79, n=14, half-wave=257.142857142857 mm, share=0.625, mixed
#: Mode 217: eigenvalue=747.25, n=14, half-wave=257.142857142857 mm, share=0.659, mixed
#: Mode 218: eigenvalue=747.25, n=14, half-wave=257.142857142857 mm, share=0.659, mixed
#: Mode 219: eigenvalue=749.44, n=14, half-wave=257.142857142857 mm, share=0.706, dominant
#: Mode 220: eigenvalue=752.54, n=36, half-wave=100.0 mm, share=0.840, dominant
#: Mode 221: eigenvalue=753.02, n=36, half-wave=100.0 mm, share=0.742, dominant
#: Mode 222: eigenvalue=753.02, n=36, half-wave=100.0 mm, share=0.742, dominant
#: Mode 223: eigenvalue=753.58, n=38, half-wave=94.7368421052632 mm, share=0.470, mixed
#: Mode 224: eigenvalue=755.24, n=2, half-wave=1800.0 mm, share=0.557, mixed
#: Mode 225: eigenvalue=755.24, n=2, half-wave=1800.0 mm, share=0.557, mixed
#: Mode 226: eigenvalue=756.01, n=36, half-wave=100.0 mm, share=0.275, mixed
#: Mode 227: eigenvalue=757.44, n=2, half-wave=1800.0 mm, share=0.796, dominant
#: Mode 228: eigenvalue=763, n=13, half-wave=276.923076923077 mm, share=0.652, mixed
#: Mode 229: eigenvalue=766.08, n=13, half-wave=276.923076923077 mm, share=0.543, mixed
#: Mode 230: eigenvalue=766.08, n=13, half-wave=276.923076923077 mm, share=0.543, mixed
#: Mode 231: eigenvalue=769.98, n=13, half-wave=276.923076923077 mm, share=0.649, mixed
#: Mode 232: eigenvalue=771.12, n=1, half-wave=3600.0 mm, share=0.890, dominant
#: Mode 233: eigenvalue=771.12, n=1, half-wave=3600.0 mm, share=0.890, dominant
#: Mode 234: eigenvalue=771.61, n=13, half-wave=276.923076923077 mm, share=0.363, mixed
#: Mode 235: eigenvalue=772.48, n=39, half-wave=92.3076923076923 mm, share=0.531, mixed
#: Mode 236: eigenvalue=772.61, n=1, half-wave=3600.0 mm, share=0.825, dominant
#: Mode 237: eigenvalue=772.61, n=1, half-wave=3600.0 mm, share=0.824, dominant
#: Mode 238: eigenvalue=773.56, n=1, half-wave=3600.0 mm, share=0.890, dominant
#: Mode 239: eigenvalue=775.5, n=13, half-wave=276.923076923077 mm, share=0.337, mixed
#: Mode 240: eigenvalue=777.96, n=12, half-wave=300.0 mm, share=0.602, mixed
#: Mode 241: eigenvalue=782.35, n=5, half-wave=720.0 mm, share=0.992, dominant
#: Mode 242: eigenvalue=782.35, n=5, half-wave=720.0 mm, share=0.992, dominant
#: Mode 243: eigenvalue=783.5, n=12, half-wave=300.0 mm, share=0.606, mixed
#: Mode 244: eigenvalue=783.5, n=12, half-wave=300.0 mm, share=0.606, mixed
#: Mode 245: eigenvalue=785.29, n=40, half-wave=90.0 mm, share=0.869, dominant
#: Mode 246: eigenvalue=785.97, n=40, half-wave=90.0 mm, share=0.496, mixed
#: Mode 247: eigenvalue=785.97, n=40, half-wave=90.0 mm, share=0.496, mixed
#: Mode 248: eigenvalue=786.13, n=40, half-wave=90.0 mm, share=0.845, dominant
#: Mode 249: eigenvalue=787.9, n=11, half-wave=327.272727272727 mm, share=0.471, mixed
#: Mode 250: eigenvalue=791.17, n=38, half-wave=94.7368421052632 mm, share=0.744, dominant
#: Written: D:\CFS-Column\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15\BU_BOLT_L3600_M20_modal_wavelengths [.csv, _spectra.csv, .png, _report.json]
#: [13:50:09] 4/4 ENHANCED: mode families, spectra, envelopes and interactive report.
#: Model: D:/CFS-Column/A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15/BU_BOLT_L3600_M20.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     4
#: Number of Meshes:             4
#: Number of Element Sets:       9
#: Number of Node Sets:          162
#: Number of Steps:              1
#: Enhanced section diagnostics: 50/250 modes
#: Enhanced section diagnostics: 100/250 modes
#: Enhanced section diagnostics: 150/250 modes
#: Enhanced section diagnostics: 200/250 modes
#: Enhanced section diagnostics: 250/250 modes
#: ENHANCED COMPLETE: 250 modes; D:\CFS-Column\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15\BU_BOLT_L3600_M20_modal_wavelengths_enhanced.html
#: [13:50:18] COMPLETE: D:\CFS-Column\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15
print('RT script done')
#: RT script done
