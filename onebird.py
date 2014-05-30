import venture.shortcuts as s

from model import *
from utils import avgFinalValue

name = "onebird"
width = 4
height = 4
cells = width * height

Y = 2
D = 20

runs=1

def run(y):
  ripl = s.make_puma_church_prime_ripl()
  
  params = {
    "name":name,
    "cells":cells,
    "years":[y],
    "days":range(D)
  }

  onebird = OneBird(ripl, params)
  
  def sweep(r, *args):
    for i in range(0):
      onebird.inferMove(ripl=r)
    #r.infer('(slice hypers one %d)' % num_features)
    r.infer('(mh hypers one %d)' % 50 * (1 + num_features))

  history, _ = onebird.runFromConditional(D, runs=runs, infer=sweep, verbose=True)
  history.hypers = [avgFinalValue(history, 'hypers%d' % k) for k in range(num_features)]
  history.save(directory="%s/%d" % (name, y))

def runInParallel():
  from multiprocessing import Process

  processes = []

  for y in range(Y):
    p = Process(target = run, args=(y,))
    processes.append(p)
    p.start()

  import pickle

  histories = []

  for y in range(Y):
    processes[y].join()
    with open("%s/%d/run_from_conditional" % (name, y)) as f:
      histories.append(pickle.load(f))

  import numpy as np

  hypers = [np.average([h.hypers[i] for h in histories]) for i in range(num_features)]
  print hypers

runInParallel()
