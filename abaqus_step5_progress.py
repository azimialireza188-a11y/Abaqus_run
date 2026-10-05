"""Small, immediately flushed progress messages for the Step5 workflow."""
import os
import sys
import time


def report(stage, message):
    """Mirror progress to stderr and, when configured, an external tail-safe log."""
    line = '[%s] [%s] %s' % (time.strftime('%H:%M:%S'), stage, message)
    try:
        sys.stderr.write(line + '\n')
        sys.stderr.flush()
    except Exception:
        pass

    path = os.environ.get('STEP5_PROGRESS_LOG', '').strip()
    if path:
        try:
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(line + '\n')
                stream.flush()
        except Exception:
            # Progress reporting must never change solver/queue behavior.
            pass


def elapsed_text(seconds):
    seconds = max(0, int(seconds))
    return '%02d:%02d:%02d' % (seconds // 3600, seconds // 60 % 60, seconds % 60)
