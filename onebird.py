import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from model import *
from venture.unit import VentureUnit

from utils import avgFinalValue

width = 4
height = 4
cells = width * height
name = "onebird"

Y = 3
D = 20

runs=3

parameters = {
  "name":name,
  "cells":cells,
  "Y":Y,
  "D":D
}

onebird = OneBird(ripl, parameters)

def sweep(r, *args):
  #for y, d in unconstrained:
  #  ripl.infer({"kernel":"gibbs", "scope":"move", "block":(y, d-1), "transitions":1})
  r.infer('(gibbs move one %d)' % 50)
  r.infer('(mh hypers one %d)' % num_features)

d="onebird-mh"
#history, _ = onebird.runConditionedFromPrior(Y * D, runs=3, infer=sweep, verbose=True)
history, _ = onebird.runFromConditional(Y * D, runs=runs, infer=sweep, verbose=True)

hypers = [avgFinalValue(h, 'hypers%d' % k) for k in range(num_features)]
history.hypers = hypers

history.save(directory=d)
#history.plotOneSeries('logscore', directory=d)
print hypers
