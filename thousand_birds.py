import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from utils import *
from model import Continuous

num_features = 4

width = 10
height = 10
cells = width * height
total_birds = 1000
name = "10x10x1000-train"

Y = 1
D = 5
years = range(Y)
days = range(D)

parameters = {
  "name":"10x10x1000-train",
  "cells":cells,
  "total_birds":total_birds,
  "years":years,
  "days":days
}

Continuous.loadAssumes(ripl, name, cells, total_birds)
Continuous.loadObserves(ripl, name, years, days)

ripl.infer("(mh default one 10)")
