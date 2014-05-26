import venture.shortcuts as s
from venture.unit import VentureUnit

from model import *
from utils import avgFinalValue

import math
from itertools import product

name = "onebird"
width = 4
height = 4
cells = width * height

Y = 1
D = 20

runs=1

def sweep(r, *args):
  #for y, d in unconstrained:
  #  ripl.infer({"kernel":"gibbs", "scope":"move", "block":(y, d-1), "transitions":1})
  r.infer('(gibbs move one %d)' % D)
  r.infer('(mh hypers one %d)' % num_features)

def run(y):
  ripl = s.make_puma_church_prime_ripl()
  
  params = {
    "name":name,
    "cells":cells,
    "years":[y],
    "days":range(D)
  }

  onebird = OneBird(ripl, params)
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


def checkHypers(ripl, hypers):
  print hypers
  for i in range(num_features):
    ripl.force('hypers%d' % i, hypers[i])
  ripl.infer('(gibbs move one %d)' % D)
  return ripl.get_global_logscore()

def gridHypers(*grid):
  ripl = s.make_puma_church_prime_ripl()
  
  params = {
    "name":name,
    "cells":cells,
    "years":[0],
    "days":range(D)
  }

  onebird = OneBird(ripl, params)
  onebird.loadAssumes()
  onebird.loadObserves()
  
  scores = {hypers:checkHypers(ripl, hypers) for hypers in grid}
  return scores

scores = gridHypers(*product(*[range(-10, 11, 4) for i in range(4)]))
scores = sorted(scores.items(), key=lambda (k,v): v, reverse=True)

