import venture.shortcuts as s
from utils import *
from venture.unit import VentureUnit
from venture.ripl.ripl import _strip_types

num_features = 4

def loadFeatures(dataset, name, years, days):
  features_file = "data/input/dataset%d/%s-features.csv" % (dataset, name)
  print "Loading features from %s" % features_file
  features = readFeatures(features_file)
  
  for (y, d, i, j) in features.keys():
    if y not in years:
      del features[(y, d, i, j)]
  
  return toVenture(features)

def loadObservations(ripl, dataset, name, years, days):
  observations_file = "data/input/dataset%d/%s-observations.csv" % (dataset, name)
  observations = readObservations(observations_file)

  for y in years:
    for (d, ns) in observations[y]:
      if d not in days: continue
      for i, n in enumerate(ns):
        #print y, d, i
        ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)

class OneBird(VentureUnit):
  
  def __init__(self, ripl, params):
    self.name = params['name']
    self.cells = params['cells']
    self.years = params['years']
    self.days = params['days']
    self.features = loadFeatures(1, self.name, self.years, self.days)
    super(OneBird, self).__init__(ripl, params)
  
  def loadAssumes(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    print "Loading assumes"

    # we want to infer the hyperparameters of a log-linear model
    for k in range(num_features):
      ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %d (normal 0 10))' % k)
    
    # the features will all be observed
    #ripl.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')
    ripl.assume('features', self.features)

    # phi is the unnormalized probability of a bird moving
    # from cell i to cell j on day d
    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp %s))))"""
       % fold('+', '(* hypers_k_ (lookup fs _k_))', '_k_', num_features))

    ripl.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
        fold('simplex', '(phi y d i j)', 'j', self.cells) +
      '))')
    
    ripl.assume('cell_array', fold('array', 'j', 'j', self.cells))
    
    # samples where a bird would move to from cell i on day d
    # the bird's id is used to identify the scope
    ripl.assume('move', """
      (lambda (y d i)
        (scope_include (quote move) (array y d)
            (categorical
              (scope_exclude (quote move)
                (get_bird_move_dist y d i))
              cell_array)))""")

    ripl.assume('get_bird_pos', """
      (mem (lambda (y d)
        (if (= d 0) 0
          (move y (- d 1) (get_bird_pos y (- d 1))))))""")

    ripl.assume('count_birds', """
      (lambda (y d i)
        (if (= (get_bird_pos y d) i)
          1 0))""")

    ripl.assume('observe_birds', '(lambda (y d i) (poisson (+ (count_birds y d i) 0.0001)))')

  def loadObserves(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
  
    observations_file = "data/input/dataset%d/%s-observations.csv" % (1, self.name)
    observations = readObservations(observations_file)

    unconstrained = []

    for y in self.years:
      for (d, ns) in observations[y]:
        if d not in self.days: continue
        if d == 0: continue
        
        loc = None
        
        for i, n in enumerate(ns):
          if n > 0:
            loc = i
            break
        
        if loc is None:
          unconstrained.append((y, d))
          #ripl.predict('(get_bird_pos %d %d)' % (y, d))
        else:
          ripl.observe('(get_bird_pos %d %d)' % (y, d), loc)
    
    return unconstrained
  
  def makeAssumes(self):
    self.loadAssumes(ripl=self)
  
  def makeObserves(self):
    self.loadObserves(ripl=self)

class Continuous(VentureUnit):

  def __init__(self, ripl, params):
    self.name = params['name']
    self.cells = params['cells']
    self.dataset = params['dataset']
    self.total_birds = params['total_birds']
    self.years = range(params['Y'])
    self.days = range(params['D'])
    self.hypers = params["hypers"]
    self.ground = readReconstruction(params)
    self.features = loadFeatures(self.dataset, self.name, self.years, self.days)
    super(Continuous, self).__init__(ripl, params)

  def loadAssumes(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    print "Loading assumes"
    
    ripl.assume('total_birds', self.total_birds)
    ripl.assume('cells', self.cells)

    #ripl.assume('num_features', num_features)

    # we want to infer the hyperparameters of a log-linear model
    for k, b in enumerate(self.hypers):
      ripl.assume('hypers%d' % k,  b)
    
    # the features will all be observed
    #ripl.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')
    ripl.assume('features', self.features)

    # phi is the unnormalized probability of a bird moving
    # from cell i to cell j on day d
    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp %s))))"""
       % fold('+', '(* hypers_k (lookup fs _k))', '_k', num_features))

    ripl.assume('get_bird_move_dist', """
      (lambda (y d i)
        (lambda (j)
          (phi y d i j)))""")
    
    ripl.assume('foldl', """
      (lambda (op x min max f)
        (if (= min max) x
          (foldl op (op x (f min)) (+ min 1) max f)))""")

    ripl.assume('multinomial_func', """
      (lambda (n min max f)
        (let ((normalize (foldl + 0 min max f)))
          (mem (lambda (i)
            (poisson (* n (/ (f i) normalize)))))))""")

    ripl.assume('count_birds', """
      (mem (lambda (y d i)
        (if (= d 0) (if (= i 0) total_birds 0)""" +
          fold('+', '(get_birds_moving y (- d 1) __j i)', '__j', self.cells) + ")))")
    
    ripl.assume('bird_movements_loc', """
      (mem (lambda (y d i)
        (let ((normalize (foldl + 0 0 cells (lambda (j) (phi y d i j)))))
          (mem (lambda (j)
            (let ((n (* (count_birds y d i) (/ (phi y d i j) normalize))))
              (if (= n 0) 0
                (scope_include d (array y d i j)
                  (poisson n)))))))))""")
    
    #ripl.assume('bird_movements', '(mem (lambda (y d) %s))' % fold('array', '(bird_movements_loc y d __i)', '__i', self.cells))
    
    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')
    
    ripl.assume('get_birds_moving', """
      (lambda (y d i j)
        ((bird_movements_loc y d i) j))""")
    
    ripl.assume('get_birds_moving1', '(lambda (y d i) %s)' % fold('array', '(get_birds_moving y d i __j)', '__j', self.cells))
    ripl.assume('get_birds_moving2', '(lambda (y d) %s)' % fold('array', '(get_birds_moving1 y d __i)', '__i', self.cells))
    ripl.assume('get_birds_moving3', '(lambda (d) %s)' % fold('array', '(get_birds_moving2 __y d)', '__y', len(self.years)))
  
  def loadObserves(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    print "Loading observations"
    loadObservations(ripl, self.dataset, self.name, self.years, self.days)
  
  def loadModel(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    self.loadAssumes(ripl)
    self.loadObserves(ripl)
  
  def makeAssumes(self):
    self.loadAssumes(ripl=self)
  
  def makeObserves(self):
    self.loadObserves(ripl=self)
  
  def updateObserves(self, d):
    self.days.append(d)
    #if d > 0: self.ripl.forget('bird_moves')
    
    loadObservations(self.ripl, self.dataset, self.name, self.years, [d])
    self.ripl.infer('(incorporate)')
    #self.ripl.predict(fold('array', '(get_birds_moving3 __d)', '__d', len(self.days)-1), label='bird_moves')
  
  def getBirdMoves(self):
    #print "Sampling bird movements"
    
    bird_moves_raw = self.ripl.report('bird_moves')
    #return ripl.sample('(get_birds_moving4)')
    
    bird_moves = {}
    
    for d in self.days[:-1]:
      for y in self.years:
        for i in range(self.cells):
          for j in range(self.cells):
            bird_moves[(y, d, i, j)] = bird_moves_raw[d][y][i][j]
    
    return bird_moves
  
  def computeScoreDay(self, d):
    bird_moves = self.ripl.sample('(get_birds_moving3 %d)' % d)
    
    score = 0
    
    for y in self.years:
      for i in range(self.cells):
        for j in range(self.cells):
          score += (bird_moves[y][i][j] - self.ground[(y, d, i, j)]) ** 2
    
    return score
  
  def computeScore(self):
    infer_bird_moves = self.getBirdMoves()

    score = 0
    
    for key in infer_bird_moves:
      score += (infer_bird_moves[key] - self.ground[key]) ** 2

    return score

