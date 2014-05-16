import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from model import *

width = 4
height = 4
cells = width * height
total_birds = 1

Y = 30
D = 20

years = range(Y)
days = range(D)

parameters = {
  "name":"onebird",
  "cells":cells,
  "total_birds":total_birds,
  "years":years,
  "days":days
}

model = BirdsModel(ripl, parameters)

def sweep(r, *args):
  r.infer('(gibbs move one %d)' % 5)
  r.infer('(mh hypers one 4)')

directory = "onebird"
history, _ = model.runFromConditional(Y * D, runs=3, infer=sweep, verbose=True)
history.save(directory = directory)
history.plot('logscore', directory = directory)
