import time

import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from venture.ripl.ripl import _strip_types

from utils import *
from model import Continuous

num_features = 4

width = 10
height = 10
cells = width * height

dataset = 3
total_birds = 1000 if dataset == 2 else 1000000
name = "%dx%dx%d-train" % (width, height, total_birds)
Y = 1
D = 3

runs = 1

# these are ground truths
hypers = [5, 10, 10, 10]

params = {
  "name":name,
  "cells":cells,
  "dataset":dataset,
  "total_birds":total_birds,
  "Y":Y,
  "D":D,
  "hypers":hypers,
}

cont = Continuous(ripl, params)

def loadFromPrior():
  cont.loadAssumes()

  print "Predicting observes"
  observes = []
  for y in range(Y):
    for d in range(D):
      for i in range(cells):
        n = ripl.predict('(observe_birds %d %d %d)' % (y, d, i))
        observes.append((y, d, i, n))
  
  return observes

def getBirdMoves():
  #print "Sampling bird movements"
  
  #return ripl.sample('(get_birds_moving4)')
  bird_moves = {}
  
  for y in range(Y):
    for d in range(D-1):
      for i in range(cells):
        for j in range(cells):
          bird_moves[(y, d, i, j)] = _strip_types(ripl.sivm.sample(['get_birds_moving'] + map(s.number, [y, d, i, j]))['value'])
  
  return bird_moves

#observes = loadFromPrior()
#true_bird_moves = getBirdMoves()

ground = readReconstruction(dataset)

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

def computeScore():
  infer_bird_moves = getBirdMoves()

  score = 0
  
  for key in infer_bird_moves:
    score += (infer_bird_moves[key] - ground[key]) ** 2

  return score

def run():
  print "Starting run"
  ripl.clear()
  cont.loadAssumes()
  cont.loadObserves()
  
  #print "Loading observations"
  #for (y, d, i, n) in observes:
  #  ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)

  print "Score: ", computeScore()
  print "pgibbs with %d particles" % p
  for y in range(Y):
    ripl.infer("(pgibbs %d ordered %d 1)" % (y, p))
  
  print "Score: ", computeScore()
  
  print "Starting mh sweeps"
  
  for i in range(Y * D):
    #print "MH step %d" % i
    #sweep(ripl)
    ripl.infer("(mh default one %d)" % 1000)
    print "Score: ", computeScore()

run()
