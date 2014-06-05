import venture.shortcuts as s

from model import *
from utils import avgFinalValue

name = "onebird"
width = 4
height = 4
cells = width * height

Y = 1
D = 3

runs=1

def run(y, days, runs, in_path=None):
  ripl = s.make_puma_church_prime_ripl()
  
  params = {
    "dataset":1,
    "name":name,
    "cells":cells,
    "years":[y],
    "days":days,
    "in_path":in_path
  }

  onebird = OneBird(ripl, params)
  
  def sweep(r, *args):
    for i in range(0):
      onebird.inferMove(ripl=r)
    onebird.inferHypers(ripl=r)

  history, _ = onebird.runFromConditional(D, runs=runs, infer=sweep, verbose=True)
  history.hypers = [avgFinalValue(history, 'hypers%d' % k) for k in range(num_features)]
  history.save(directory="%s/%d" % (name, y))

def runInParallel(Y=Y, D=D, runs=runs, path=None, **kwargs):
  from multiprocessing import Process

  processes = []

  for y in range(Y):
    p = Process(target = run, args=(y,range(D),runs,path))
    processes.append(p)
    p.start()

  for y in range(Y):
    processes[y].join()

def computeHypers(Y=Y, path=None, **kwargs):
  histories = []

  import pickle
  
  for y in range(Y):
    with open("%s/%d/run_from_conditional" % (name, y)) as f:
      histories.append(pickle.load(f))
  
  from numpy import average, std
  
  hypers = zip(*[h.hypers for h in histories])
  
  means = map(average, hypers)
  stds = map(std, hypers)
  
  print "Means: " + str(means)
  print "Stds:  " + str(stds)
  
  writeHypers(means, path=path, dataset=1)

def doOneBird(inPath=None, outPath=None, **kwargs):
  runInParallel(path=inPath, **kwargs)
  computeHypers(path=outPath, **kwargs)

if __name__ == "__main__":
  doOneBird()
