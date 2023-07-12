import time


def timeit(func):
  def wrapper(*args, **kwargs):
    t0 = time.time()
    ans = func(*args, **kwargs)
    dt = time.time() - t0
    return ans, dt
  return wrapper

class Timmer(object):
  def __init__(self, name) -> None:
    super().__init__()
    self.name = name
    self.start = 0.
    self.stop = 0.
    self.start = time.time()
  
  def end(self):
    self.stop = time.time()

  def result(self):
    return self.stop - self.start

  def __str__(self):
    #fmt = '%Y-%m-%d-%H.%M.%S'
    return '|Timmer %s| %.3f sec' % (self.name, self.result())


class Timmers(object):
  def __init__(self, f) -> None:
    super().__init__()
    self.f = f
    #assert os.path.exists(f)
    self.timmers = []
    self.name2idx = {}

  def add_timmer(self, name):
    self.timmers.append(Timmer(name))
    idx = len(self.timmers) - 1
    self.name2idx[name] = idx
  
  def end_timmer(self, name):
    idx = self.name2idx[name]
    self.timmers[idx].end()
  
  def __str__(self):
    msg = ''
    for t in self.timmers:
      msg += '%s\n' % t.__str__()
    return msg
  
  def save(self):
    with open(self.f, 'w') as h:
      h.write('name\tduration(sec)\n')
      [h.write('%s\t%.3f\n' % (t.name, t.result())) for t in self.timmers]


def ttest_timmer():
  ta, tb = Timmer('A'), Timmer('B')
  time.sleep(3.2)
  ta.end()
  time.sleep(1.2)
  tb.end()
  print(ta)
  print(tb)


def ttest_timmers():
  f = 'test_timmer.txt'
  ts = Timmers(f)
  [ts.add_timmer('a%d' % i) for i in range(10)]
  for i in range(10):
    time.sleep(0.1)
    ts.end_timmer('a%d' % i)
  print(ts)
  ts.save()


if __name__ == '__main__':
  ttest_timmers()