import venture.shortcuts as s

ripl = s.make_puma_church_prime_ripl()

from model import *

width = 4
height = 4
cells = width * height
total_birds = 1

years = range(1, 2)
days = range(1, 21)

model = Model(ripl, "onebird", cells, total_birds, years, days)
