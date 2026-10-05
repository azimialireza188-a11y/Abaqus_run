# -*- coding: utf-8 -*-
"""Explicit Abaqus/CAE noGUI entry script; do not import as a library.

CAE can execute scripts in a namespace other than __main__. This launcher
therefore calls the reusable queue main explicitly, without a name guard.
"""
import os
import sys

SCRIPT_PATH = os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename))
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from abaqus_step5_progress import report
report('DRIVER', 'STEP5 NOGUI DRIVER STARTED')
from abaqus_step5_queue import main

try:
    main()
except Exception as error:
    report('QUEUE FAILED', str(error))
    raise
