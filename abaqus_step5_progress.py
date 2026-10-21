"""Small, immediately flushed progress messages for the Step5 workflow."""
import os
import sys
import time
import threading
from contextlib import contextmanager


def report(stage, message):
    """Mirror progress to stderr and, when configured, an external tail-safe log."""
    line = '[%s] [%s] %s' % (time.strftime('%H:%M:%S'), stage, message)
    delivered = False
    path = os.environ.get('STEP5_PROGRESS_LOG', '').strip()
    if path:
        try:
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(line + '\n')
                stream.flush()
            delivered = True
        except Exception:
            # Progress reporting must never change solver/queue behavior.
            pass
    if not (delivered and os.environ.get('STEP5_PROGRESS_LOG_ONLY') == '1'):
        try:
            sys.stderr.write(line + '\n')
            sys.stderr.flush()
        except Exception:
            pass


@contextmanager
def console_bridge(directory, environment):
    """Tail CAE's flushed log in the original terminal, even when CAE hides stdout."""
    environment = dict(environment)
    environment['STEP5_REPORT_PYTHON'] = os.path.abspath(sys.executable)
    environment['STEP5_REPORT_PATH'] = environment.get('PATH', '')
    path = os.path.join(os.path.abspath(directory), 'STEP5_console_progress.log')
    # Append to the existing log, but do not replay an earlier invocation.
    with open(path, 'ab') as stream:
        offset = stream.tell()
    environment.update(STEP5_PROGRESS_LOG=path, STEP5_PROGRESS_LOG_ONLY='1')
    stopped = threading.Event()

    def follow():
        with open(path, 'rb') as stream:
            stream.seek(offset)
            pending = b''
            while True:
                pending += stream.read()
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    print(line.decode('utf-8', errors='replace').rstrip('\r'), flush=True)
                if stopped.is_set():
                    if pending:
                        print(pending.decode('utf-8', errors='replace'), flush=True)
                    break
                stopped.wait(.2)

    thread = threading.Thread(target=follow, name='step5-console', daemon=True)
    thread.start()
    try:
        yield environment
    finally:
        stopped.set()
        thread.join()


def elapsed_text(seconds):
    seconds = max(0, int(seconds))
    return '%02d:%02d:%02d' % (seconds // 3600, seconds // 60 % 60, seconds % 60)
