import venture.shortcuts as s
from utils import *
from venture.unit import VentureUnit

num_features = 4

def loadFeatures(ripl, dataset, name, years, days):
  features_file = "data/input/dataset%d/%s-features.csv" % (dataset, name)
  features = readFeatures(features_file)
  
  for (y, d, i, j) in features.keys():
    if y not in years or d not in days:
      del features[(y, d, i, j)]
  
  ripl.assume('features', toVenture(features))

def loadObservations(ripl, dataset, name, years, days):
  observations_file = "data/input/dataset%d/%s-observations.csv" % (dataset, name)
  observations = readObservations(observations_file)

  for y in years:
    for (d, ns) in observations[y]:
      if d not in days: continue
      for i, n in enumerate(ns):
        print y, d, i
        ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)

class OneBird(VentureUnit):
  
  def __init__(self, ripl, params):
    self.name = params['name']
    self.cells = params['cells']
    self.years = range(params['Y'])
    self.days = range(params['D'])
    super(OneBird, self).__init__(ripl, params)
  
  def loadAssumes(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    # we want to infer the hyperparameters of a log-linear model
    for k in range(num_features):
      ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %d (normal 0 10))' % k)
    
    # the features will all be observed
    #ripl.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')
    loadFeatures(ripl, 1, self.name, self.years, self.days)

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
          (move y (- d 1) (get_bird_pos y (- d 1)))
        )
      ))
    """)

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
    loadFeatures(ripl, self.dataset, self.name, self.years, self.days)

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

    ripl.assume('clamp_min', '(lambda (min x) (biplex (< x min) min x))')

    ripl.assume('approx_binomial', """
      (lambda (n p)
        (clamp_min 0
          (normal (* p n) (sqrt (* n (- p (* p p)))))))""")

    ripl.assume('multinomial_func', """
      (lambda (n min max f)
        (let ((normalize (foldl + 0 min max f)))
          (mem (lambda (i)
            (approx_binomial n (/ (f i) normalize))))))""")    

    ripl.assume('count_birds', """
      (mem (lambda (y d i)
        (if (= d 0)
          (if (= i 0) total_birds 0)
          (let ((moves (bird_movements y (- d 1))))
            (scope_include y (- d 1)""" +
              fold('+', '((lookup moves _j) i)', '_j', self.cells) + ')))))')
    
    ripl.assume('bird_movements_loc', """
      (mem (lambda (y d i)
        (multinomial_func (count_birds y d i) 0 cells (get_bird_move_dist y d i))))""")
    
    ripl.assume('bird_movements', '(mem (lambda (y d) %s))' % fold('array', '(bird_movements_loc y d _i_)', '_i_', self.cells))
    
    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')
    
    ripl.assume('get_birds_moving', """
      (lambda (y d i j)
        ((bird_movements_loc y d i) j))""")
    
    #ripl.assume('get_birds_moving1', '(lambda (y d i) %s)' % fold('array', '(get_birds_moving y d i _j)', '_j', cells))
    #ripl.assume('get_birds_moving2', '(lambda (y d) %s)' % fold('array', '(get_birds_moving1 y d _i)', '_i', cells))
    #ripl.assume('get_birds_moving3', '(lambda (y) %s)' % fold('array', '(get_birds_moving2 y _d)', '_d', params["D"]-1))
    #ripl.assume('get_birds_moving4', '(lambda () %s)' % fold('array', '(get_birds_moving3 _y)', '_y', params["Y"]))
  
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

