import time

import venture.shortcuts as s

from utils import *
from model import Poisson, num_features

width = 10
height = 10
cells = width * height

# these are ground truths
hypers = [5, 10, 10, 10]

def run(verbose=True, Y=1, D=2, dataset=2, in_path=None, out_path=None, steps=2):
  ripl = s.make_puma_church_prime_ripl()

  total_birds = 1000 if dataset == 2 else 1000000
  name = "%dx%dx%d-test" % (width, height, total_birds)

  params = {
    "name":name,
    "width":width,
    "height":height,
    "cells":cells,
    "dataset":dataset,
    "total_birds":total_birds,
    "years":range(Y),
    "days":[],
    "hypers":hypers,
    "in_path":in_path,
    "out_path":out_path,
  }

  model = Poisson(ripl, params)

  if verbose:
    print "Starting run"
  ripl.clear()
  model.loadAssumes()
  
  predictions = {}
  
  for d in range(D-1):
    if verbose:
      print "Day %d" % d
    model.updateObserves(d)
    
    if d > 0:
      for i in range(5):
        ripl.infer({"kernel":"mh", "scope":d-1, "block":"one", "transitions": Y*1000})
    
    last_day = (d == D-2)
    
    if last_day:
      bird_moves = ripl.predict('(get_birds_moving3 %d)' % d, label="predict%d" % d)
    else:
      bird_moves = [ripl.predict('(get_birds_moving3 %d)' % t, label="predict%d" % t) for t in [d, d+1]]
    
    for y in range(Y):
      for i in range(cells):
        for j in range(cells):
          if last_day:
            predictions[(y, d-1, i, j)] = [bird_moves[y][i][j], -1]
          else:
            predictions[(y, d-1, i, j)] = [b[y][i][j] for b in bird_moves]
    
    for t in ([d] if last_day else [d, d+1]):
      ripl.forget("predict%d" % t)
    
  #model.drawBirdLocations()
  
  writePredictions(predictions, **params)

if __name__ == "__main__":
  run()
