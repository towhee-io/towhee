import threading
import time


def run(runner):
    runner.process()


def start_runner(runner):
    t = threading.Thread(target=run, args=(runner, ))
    t.start()
    time.sleep(0.01) # make sure runner started.
    return t
