import venture.shortcuts as s
from venture.unit import VentureUnit

from model import *
from utils import avgFinalValue

import math

name = "onebird"
width = 4
height = 4
cells = width * height

Y = 3
D = 20

runs=1

def sweep(r, *args):
  #for y, d in unconstrained:
  #  ripl.infer({"kernel":"gibbs", "scope":"move", "block":(y, d-1), "transitions":1})
  r.infer('(gibbs move one %d)' % (D / math.e))
  r.infer('(slice hypers one %d)' % num_features)

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

