"""Small, immediately flushed console messages for the Step5 workflow."""
import sys
import time


def report(stage, message):
    """Write progress to stderr because Abaqus/CAE noGUI may suppress stdout."""
    line = '[%s] [%s] %s\n' % (time.strftime('%H:%M:%S'), stage, message)
    sys.stderr.write(line)
    sys.stderr.flush()


def elapsed_text(seconds):
    seconds = max(0, int(seconds))
    return '%02d:%02d:%02d' % (seconds // 3600, seconds // 60 % 60, seconds % 60)
