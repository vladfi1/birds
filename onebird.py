import venture.shortcuts as s

from model import *
from utils import avgFinalValue

name = "onebird"
width = 4
height = 4
cells = width * height

Y = 3
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
    onebird.inferHypers(ripl=r)

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

  for y in range(Y):
    processes[y].join()

def computeWeightedHypers(years=range(Y)):
  lxs = [[] for k in range(num_features)]

  import pickle
  
  for y in years:
    with open("%s/%d/run_from_conditional" % (name, y)) as f:
      history = pickle.load(f)
      logscores = history.nameToSeries['logscore']
      for k in range(num_features):
        for lseries, xseries in zip(logscores, history.nameToSeries['hypers%d' % k]):
          lxs[k].append((lseries.values[-1], xseries.values[-1]))
  
  hypers = map(weightedAverage, lxs)
  print hypers
  
  writeHypers(hypers, dataset=1)

def computeHypers(years=range(Y)):
  histories = []

  import pickle
  
  for y in years:
    with open("%s/%d/run_from_conditional" % (name, y)) as f:
      histories.append(pickle.load(f))
  
  from numpy import average, std
  
  hypers = zip(*[h.hypers for h in histories])
  
  means = map(average, hypers)
  stds = map(std, hypers)
  
  print "Means: " + str(means)
  print "Stds:  " + str(stds)
  
  writeHypers(means, dataset=1)

if __name__ == "__main__":
  runInParallel()
  computeHypers()
