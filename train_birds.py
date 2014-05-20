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

Y = 1
D = 2

runs = 1

# these are estimates from onebird
hypers = [1.7223437017981613, 3.4445635302098445, 2.5864910484370607, 3.5926413162397104]

params = {
  "name":name,
  "cells":cells,
  "total_birds":total_birds,
  "Y":Y,
  "D":D,
  "hypers":hypers,
}

#thousand_birds = Continuous(ripl, parameters)

Continuous.loadAssumes(ripl, **params)

print "Predicting observes"
observes = []
for y in range(Y):
  for d in range(D):
    for i in range(cells):
      n = ripl.predict('(observe_birds %d %d %d)' % (y, d, i))
      observes.append((y, d, i, n))

def getBirdMoves():
  print "Sampling bird movements"
  
  #return ripl.sample('(get_birds_moving4)')
  bird_moves = []
  
  for y in range(Y):
    for d in range(D-1):
      for i in range(cells):
        for j in range(cells):
          bird_moves.append(_strip_types(ripl.sivm.sample(['get_birds_moving'] + map(s.number, [y, d, i, j]))['value']))
  
  return bird_moves

true_bird_moves = getBirdMoves()

import multiprocessing
p = multiprocessing.cpu_count() / 2

print "Using %d particles" % p

def sweep(r, *args):
  #for y in range(Y):
  #  r.infer("(pgibbs %d ordered %d 1)" % (y, p))
  for y in range(Y):
    #r.infer("(mh %d one %d)" % (y, 1))
    r.infer("(mh default one %d)" % cells)

def computeScore():
  infer_bird_moves = getBirdMoves()

  score = 0

  for n1, n2 in zip(true_bird_moves, infer_bird_moves):
    score += (n1 - n2) ** 2

  return score

def run():
  print "Starting run"
  ripl.clear()
  Continuous.loadAssumes(ripl, **params)
  
  print "Loading observations"
  for (y, d, i, n) in observes:
    ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)

  for i in range(Y * D):
    print "Inference step %d" % i
    sweep(ripl)
    print computeScore()

run()
