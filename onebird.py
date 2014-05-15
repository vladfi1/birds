import venture.shortcuts as s

ripl = s.make_puma_church_prime_ripl()

from model import *

width = 4
height = 4
cells = width * height
total_birds = 1

Y = 3
D = 20

years = range(1, Y+1)
days = range(1, D+1)

model = Model(ripl, "onebird", cells, total_birds, years, days)
print "From prior: ", model.likelihood()

model.inferMove(10 * Y * D)
print "Gibbs on birds: ", model.likelihood()

def sweep():
  model.inferMove(5 * Y)
  model.inferHypers(20)
  print model.likelihood()

for i in range(200):
  sweep()
