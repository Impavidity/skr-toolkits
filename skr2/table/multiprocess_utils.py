from multiprocessing import Value, Lock
import datetime

class AtomicCounter(object):
  def __init__(self, print_every=1000):
    self.value = Value("i", 0)
    self.lock = Lock()
    self.print_every = print_every

  def increment(self):
    with self.lock:
      self.value.value += 1
      if self.value.value % self.print_every == 0:
        print("[{}] {} examples processed.".format(datetime.datetime.now(), self.value.value))
      return self.value.value
