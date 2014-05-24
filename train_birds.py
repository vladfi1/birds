import time

import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from utils import *
from model import Poisson

num_features = 4

width = 10
height = 10
cells = width * height

dataset = 2
total_birds = 1000 if dataset == 2 else 1000000
name = "%dx%dx%d-train" % (width, height, total_birds)
Y = 1
D = 3

runs = 1

# these are ground truths
hypers = [5, 10, 10, 10]

params = {
  "name":name,
  "width":width,
  "height":height,
  "cells":cells,
  "dataset":dataset,
  "total_birds":total_birds,
  "years":range(Y),
  "days":[0],
  "hypers":hypers,
}

model = Poisson(ripl, params)

def loadFromPrior():
  model.loadAssumes()

  print "Predicting observes"
  observes = []
  for y in range(Y):
    for d in range(D):
      for i in range(cells):
        n = ripl.predict('(observe_birds %d %d %d)' % (y, d, i))
        observes.append((y, d, i, n))
  
  return observes

#observes = loadFromPrior()
#true_bird_moves = getBirdMoves()

import multiprocessing
#p = multiprocessing.cpu_count() / 2
p = 2

print "Using %d particles" % p

def sweep(r, *args):
  t0 = time.time()
  for y in range(Y):
    r.infer("(pgibbs %d ordered %d 1)" % (y, p))
  
  t1 = time.time()
  #for y in range(Y):
    #r.infer("(mh %d one %d)" % (y, 1))
  r.infer("(mh default one %d)" % 1000)
  
  t2 = time.time()
  
  print "pgibbs: %f, mh: %f" % (t1-t0, t2-t1)

def run(pgibbs=True):
  print "Starting run"
  ripl.clear()
  model.loadAssumes()
  model.updateObserves(0)

  #model.loadObserves()
  #ripl.infer('(incorporate)')
  
  #print "Loading observations"
  #for (y, d, i, n) in observes:
  #  ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)
  
  logs = []
  t = [time.time()] # python :(
  
  def log():
    dt = time.time() - t[0]
    logs.append((ripl.get_global_logscore(), model.computeScoreDay(model.days[-2]), dt))
    print logs[-1]
    t[0] += dt
  
  for d in range(1, D):
    print "Day %d" % d
    model.updateObserves(d)
    log()
    
    for i in range(1):
      ripl.infer({"kernel":"mh", "scope":d-1, "block":"one", "transitions": Y * cells ** 2})
      log()
    
  writeReconstruction(params, model.getBirdMoves())
  
  drawBirdMoves(params, model.getBirdMoves(), 'temp')
  
  return logs

history = None
if __name__ == "__main__":
  history = run()

