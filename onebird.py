import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from model import *
from venture.unit import VentureUnit

width = 4
height = 4
cells = width * height
name = "onebird"

Y = 30
D = 20

years = range(Y)
days = range(D)

parameters = {
  "name":name,
  "cells":cells,
  "years":years,
  "days":days
}

onebird = OneBird(ripl, parameters)

def sweep(r, *args):
  #for y, d in unconstrained:
  #  ripl.infer({"kernel":"gibbs", "scope":"move", "block":(y, d-1), "transitions":1})
  r.infer('(gibbs move one %d)' % 5)
  r.infer('(mh hypers one %d)' % num_features)

directory = "onebird/"
history, _ = onebird.runFromConditional(Y * D, runs=3, infer=sweep, verbose=True)
history.save(directory = directory)
history.plotOneSeries('logscore', directory = directory)

