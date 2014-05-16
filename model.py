import venture.shortcuts as s
from utils import *
from venture.unit import VentureUnit

num_features = 4

class BirdsModel(VentureUnit):
  def makeAssumes(self):
    name = self.parameters["name"]
    cells = self.parameters["cells"]
    total_birds = self.parameters["total_birds"]
    
    #ripl.assume('width', width)
    #ripl.assume('height', height)
    #ripl.assume('cells', cells)

    #ripl.assume('square', '(lambda (x) (* x x))')
    #ripl.assume('dist_max2', '18')

    #ripl.assume('cell2X', '(lambda (cell) (int_div cell height))')
    #ripl.assume('cell2Y', '(lambda (cell) (int_mod cell height))')
    #ripl.assume('cell2P', '(lambda (cell) (make_pair (cell2X cell) (cell2Y cell)))')
    #ripl.assume('XY2cell', '(lambda (x y) (to_atom (+ (* height x) y)))')

    #ripl.assume('dist2', """
    #  (lambda (x1 y1 x2 y2)
    #    (+ 
    #      (square (- x1 x2))
    #      (square (- y1 y2))
    #    )
    #  )
    #""")

    #ripl.assume('cell_dist2', """
    #  (lambda (i j)
    #    (dist2
    #      (cell2X i) (cell2Y i)
    #      (cell2X j) (cell2Y j)
    #    )
    #  )
    #""")

    #self.assume('num_features', num_features)

    # we want to infer the hyperparameters of a log-linear model
    for k in range(num_features):
      self.assume('hypers%d' % k, '(scope_include (quote hypers) %d (normal 0 10))' % k)
    
    # the features will all be observed
    #self.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')
    
    #self.assume('fold', """
    #  (lambda (op f len)
    #    (if (= len 1) (f 0)
    #      (op
    #        (f (- len 1))
    #        (fold op f (- len 1))
    #      )
    #    )
    #  )
    #""")

    # phi is the unnormalized probability of a bird moving
    # from cell i to cell j on day d
    self.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp %s))))"""
       % fold('+', '(* hypers_k_ (lookup fs _k_))', '_k_', num_features))

    self.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
        fold('simplex', '(phi y d i j)', 'j', cells) +
      '))')
    
    self.assume('cell_array', fold('array', 'j', 'j', cells))
    
    # samples where a bird would move to from cell i on day d
    # the bird's id is used to identify the scope
    self.assume('move', """
      (lambda (y d i id)
        (scope_include (quote move) (array y d id)
        (scope_include y d
            (categorical
              (scope_exclude (quote move)
              (scope_exclude y (get_bird_move_dist y d i)))
              cell_array))))""")

    self.assume('get_bird_pos', """
      (mem (lambda (y d id)
        (if (= d 0) 0
          (move y (- d 1) (get_bird_pos y (- d 1) id) id)
        )
      ))
    """)

    #self.assume('total_birds', total_birds)

    self.assume('count_birds', '(lambda (y d i) %s)' % fold('+', '(if (= (get_bird_pos y d id) i) 1 0)', 'id', total_birds))

    self.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')

    self.loadFeatures()
  
  def makeObserves(self):
    self.loadObservations()

  def loadFeatures(self):
    features_file = "release/%s-features.csv" % self.parameters["name"]
    features = readFeatures(features_file)
    self.assume('features', toVenture(features))

  def loadObservations(self):
    observations_file = "release/%s-observations.csv" % self.parameters["name"]
    observations = readObservations(observations_file)

    for y in self.parameters["years"]:
      for (d, ns) in observations[y]:
        if d not in self.parameters["days"]: continue
        for i, n in enumerate(ns):
          print y, d, i
          self.observe('(observe_birds %d %d %d)' % (y, d, i), n)
    
    #self.ripl.infer('(incorporate)')

