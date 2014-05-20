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

years = range(Y)
days = range(D)

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
  r.infer('(slice hypers one %d)' % num_features)

d="slice"
#history, _ = onebird.runConditionedFromPrior(Y * D, runs=3, infer=sweep, verbose=True)
history, _ = onebird.runFromConditional(Y * D / 6, runs=3, infer=sweep, verbose=True)

hypers = [[history.nameToSeries['hypers%d' % k][r].values[-1] for k in range(4)] for r in range(3)]
history.hypers = hypers

history.save(directory=d)
#history.plotOneSeries('logscore', directory=d)
print [avgFinalValue(h, 'hypers%d' % k) for k in range(num_features)]
