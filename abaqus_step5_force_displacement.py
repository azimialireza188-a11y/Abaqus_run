# -*- coding: utf-8 -*-
"""Abaqus/CAE entry script for per-job force-shortening extraction.

Run via noGUI with --cae, --odb and optional --allow-partial.
Import abaqus_step5_force_displacement_core for the reusable Python API.
"""
import os
import sys
SCRIPT_PATH = os.path.abspath(
    globals().get('__file__',sys._getframe().f_code.co_filename))
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0,SCRIPT_DIR)
print('STEP5 FORCE-DISPLACEMENT EXTRACTOR STARTED')
sys.stdout.flush()
from abaqus_step5_force_displacement_core import main
main()
