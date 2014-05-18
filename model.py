import venture.shortcuts as s
from utils import *
from venture.unit import VentureUnit

num_features = 4

def loadFeatures(ripl, name):
  features_file = "release/%s-features.csv" % name
  features = readFeatures(features_file)
  ripl.assume('features', toVenture(features))

def loadObservations(ripl, name, years, days):
  observations_file = "release/%s-observations.csv" % name
  observations = readObservations(observations_file)

  for y in years:
    for (d, ns) in observations[y]:
      if d not in days: continue
      for i, n in enumerate(ns):
        print y, d, i
        ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)

class OneBird(VentureUnit):
  @staticmethod
  def loadAssumes(ripl, name, cells):
    # we want to infer the hyperparameters of a log-linear model
    for k in range(num_features):
      ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %d (normal 0 10))' % k)
    
    # the features will all be observed
    #ripl.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')
    loadFeatures(ripl, name)

    # phi is the unnormalized probability of a bird moving
    # from cell i to cell j on day d
    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp %s))))"""
       % fold('+', '(* hypers_k_ (lookup fs _k_))', '_k_', num_features))

    ripl.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
        fold('simplex', '(phi y d i j)', 'j', cells) +
      '))')
    
    ripl.assume('cell_array', fold('array', 'j', 'j', cells))
    
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

  @staticmethod
  def loadObserves(ripl, name, years, days):
    observations_file = "release/%s-observations.csv" % name
    observations = readObservations(observations_file)

    unconstrained = []

    for y in years:
      for (d, ns) in observations[y]:
        if d not in days: continue
        if d == 0: continue
        
        loc = None
        
        for i, n in enumerate(ns):
          if n > 0:
            loc = i
            break
        
        if loc is None:
          unconstrained.append((y, d))
          ripl.predict('(get_bird_pos %d %d)' % (y, d))
        else:
          ripl.observe('(observe_birds %d %d %d)' % (y, d, loc), 1)
          ripl.infer({"kernel":"gibbs", "scope":"move", "block":(y, d-1), "transitions":1})
    
    return unconstrained
  
  def makeAssumes(self):
    OneBird.loadAssumes(self, self.parameters["name"], self.parameters["cells"])
  
  def makeObserves(self):
    loadObservations(self, self.parameters["name"], self.parameters["years"], self.parameters["days"])
  
class Continuous:
  @staticmethod
  def loadAssumes(ripl, name, cells, total_birds):
    ripl.assume('total_birds', total_birds)
    ripl.assume('cells', cells)

    #ripl.assume('num_features', num_features)

    # we want to infer the hyperparameters of a log-linear model
    for k in range(num_features):
      ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %d (normal 0 10))' % k)
    
    # the features will all be observed
    #ripl.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')
    loadFeatures(ripl, name)

    # phi is the unnormalized probability of a bird moving
    # from cell i to cell j on day d
    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp %s))))"""
       % fold('+', '(* hypers_k_ (lookup fs _k_))', '_k_', num_features))

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
          (if (= i 0) total_birds 0)""" +
          fold('+', '((bird_movements y (- d 1) j) i)', 'j', cells) + ')))')
    
    ripl.assume('bird_movements', """
      (mem (lambda (y d i)
        (multinomial_func (count_birds y d i) 0 cells (get_bird_move_dist y d i))))""")
    
    ripl.assume('get_birds_moving', """
      (lambda (y d i j)
        ((bird_movements y d i) j))""")

    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')
  
  @staticmethod
  def loadObserves(ripl, name, days, years):
    loadObservations(ripl, name, days, years)
