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
hypers = [1.011776136710042, 3.315465986337299, 2.6465177320342272, 3.6932748455768913]

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

observes = {}

for y in range(Y):
  for d in range(D):
    for i in range(cells):
      observes[(y, d, i)] = ripl.predict('(observe_birds %d %d %d)' % (y, d, i))

#true_bird_moves = ripl.predict('(get_birds_moving4)')

def getBirdMoves():
  bird_moves = []
  for y in range(Y):
    for d in range(D-1):
      for i in range(cells):
        for j in range(cells):
          print y, d, i, j
          bird_moves.append(_strip_types(ripl.sivm.core_sivm.engine.predict([s.symbol('get_birds_moving')] + map(s.number, [y, d, i, j]))[1]))
  
  return bird_moves

true_bird_moves = getBirdMoves()

def sweep(r, *args):
  for y in range(Y):
    r.infer("(pgibbs %d ordered 2 1)" % y)
  r.infer("(mh default one %d)" % (5 * cells))

ripl.clear()
Continuous.loadAssumes(ripl, **params)

for (y, d, i), n in observes.iteritems():
  ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)

for i in range(Y * D):
  print i
  sweep(ripl)

#infer_bird_moves = ripl.predict('(get_birds_moving4)')

infer_bird_moves = getBirdMoves()

score = 0

for n1, n2 in zip(true_bird_moves, infer_bird_moves):
  score += (n1 - n2) ** 2

print score
