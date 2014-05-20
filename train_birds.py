import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from venture.ripl.ripl import _strip_types

from utils import *
from model import Continuous

num_features = 4

width = 10
height = 10
cells = width * height

total_birds = 1000
name = "%dx%dx%d-test" % (width, height, total_birds)
dataset = 2
Y = 1
D = 2

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
  print "Sampling bird movements"
  
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
p = multiprocessing.cpu_count() / 2

print "Using %d particles" % p

def sweep(r, *args):
  for y in range(Y):
    r.infer("(pgibbs %d ordered %d 1)" % (y, p))
  #for y in range(Y):
    #r.infer("(mh %d one %d)" % (y, 1))
  r.infer("(mh default one %d)" % (cells ** 2))

def computeScore():
  infer_bird_moves = getBirdMoves()

  score = 0
  
  for key in infer_bird_moves:
    if key in ground:
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

  for i in range(Y * D):
    print "Inference step %d" % i
    sweep(ripl)
    print computeScore()

run()
