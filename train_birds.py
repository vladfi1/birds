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
  name = "%dx%dx%d-train" % (width, height, total_birds)

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
  model.updateObserves(0)
  
  logs = []
  t = [time.time()] # python :(
  
  def log():
    dt = time.time() - t[0]
    #logs.append((ripl.get_global_logscore(), model.computeScoreDay(model.days[-2]), dt))
    logs.append((ripl.get_global_logscore(), dt))
    if verbose:
      print logs[-1]
    t[0] += dt
  
  for d in range(1, D):
    if verbose:
      print "Day %d" % d
    model.updateObserves(d)
    log()
    
    for i in range(steps):
      ripl.infer({"kernel":"mh", "scope":d-1, "block":"one", "transitions": Y*1000})
      log()
      continue
      bird_locs = model.getBirdLocations(days=[d])
      
      for y in range(Y):
        path = 'bird_moves%d/%d/%02d/' % (dataset, y, d)
        ensure(path)
        drawBirds(bird_locs[y][d], path + '%02d.png' % i, **params)
  
  #model.drawBirdLocations()
  model.writeBirdMoves()
  
  return logs

history = None
if __name__ == "__main__":
  history = run()

