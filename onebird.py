import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from model import *

width = 4
height = 4
cells = width * height
name = "onebird"

Y = 2
D = 10

years = range(Y)
days = range(D)

parameters = {
  "name":name,
  "cells":cells,
  "years":years,
  "days":days
}

OneBird.loadAssumes(ripl, name, cells)
OneBird.loadObserves(ripl, name, years, days)

#model = BirdsModel(ripl, parameters)

def sweep(r, *args):
  r.infer('(mh move one %d)' % 5)
  #r.infer('(mh hypers one 4)')

for i in range(10):
  sweep(ripl)

"""
directory = "onebird"
history, _ = model.runFromConditional(Y * D, runs=3, infer=sweep, verbose=True)
history.save(directory = directory)
history.plotOneSeries('logscore', directory = directory)
"""
