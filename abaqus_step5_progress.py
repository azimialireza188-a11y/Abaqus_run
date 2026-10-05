"""Small, immediately flushed console messages for the Step5 workflow."""
import time


def report(stage, message):
    print('[%s] [%s] %s' % (time.strftime('%H:%M:%S'), stage, message), flush=True)


def elapsed_text(seconds):
    seconds = max(0, int(seconds))
    return '%02d:%02d:%02d' % (seconds // 3600, seconds // 60 % 60, seconds % 60)
