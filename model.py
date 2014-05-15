import venture.shortcuts as s
from utils import *

num_features = 4

class Model(object):
  def __init__(self, ripl, name, cells, total_birds, years, days):
  
    self.ripl = ripl
    self.name = name
    self.cells = cells
    self.total_birds = total_birds
    self.years = years
    self.days = days
    
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

    #ripl.assume('num_features', num_features)

    # we want to infer the hyperparameters of a log-linear model
    ripl.assume('hypers', """
      (mem (lambda (k)
        (scope_include
          (quote hypers) k
          (normal 0 10)
        )
      ))
    """)

    # the features will all be observed
    ripl.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')

    #ripl.assume('fold', """
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
    ripl.assume('phi', "(mem (lambda (y d i j) (exp %s)))" % fold('+', '(* (hypers k) (features y d i j k))', 'k', num_features))

    ripl.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
        '(simplex ' + ' '.join(['(phi y d i atom<%d>)' % j for j in range(cells)]) + ')' +
      '))')

    # samples where a bird would move to from cell i on day d
    # the bird's id is used to identify the scope
    ripl.assume('move', """
      (lambda (y d i id)
        (scope_include (quote move) (array y d id)
        (scope_include y d
            (categorical
              (scope_exclude (quote move)
              (scope_exclude y
                (get_bird_move_dist y d i)
              ))
            )
        ))
      )
    """)

    ripl.assume('get_bird_pos', """
      (mem (lambda (y d id)
        (if (= d 1) atom<0>
          (move y (- d 1) (get_bird_pos y (- d 1) id) id)
        )
      ))
    """)

    #ripl.assume('total_birds', total_birds)

    ripl.assume('count_birds', '(lambda (y d i) %s)' % fold('+', '(if (= (get_bird_pos y d id) i) 1 0)', 'id', total_birds))

    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')

    self.loadFeatures()
    self.loadObservations()
    
    print ripl.sivm.core_sivm.engine.get_entropy_info()

  def loadFeatures(self):
    features_file = "release/%s-features.csv" % self.name

    features = readFeatures(features_file)
    
    for (y, d, i, j), fs in features.iteritems():
      if y not in self.years: continue
      if d not in self.days: continue
      
      i -= 1
      j -= 1
      
      for k, f in enumerate(fs):
        self.ripl.sivm.observe(['features', s.number(y), s.number(d), s.atom(i), s.atom(j), s.number(k)], s.number(f))
    
    self.ripl.infer('(incorporate)')
    self.features_likelihood = self.ripl.get_global_logscore()

  def loadObservations(self):
    observations_file = "release/%s-observations.csv" % self.name
    observations = readObservations(observations_file)

    for y in self.years:
      for (d, ns) in observations[y]:
        if d not in self.days: continue
        for i, n in enumerate(ns):
          print y, d, i
          self.ripl.observe('(observe_birds %d %d atom<%d>)' % (y, d, i), n)
    
    self.ripl.infer('(incorporate)')

  def getBirds(self, y, d):
    return [self.ripl.sample('(count_birds %d %d atom<%d>)' % (y, d, i)) for i in range(self.cells)]

  def printYear(self, y):
    for d in self.days:
      print self.getBirds(y, d)

  def getBird(self, y, id):
    return [self.ripl.sample('(get_bird_pos %d %d %d)' % (y, d, id)) for d in self.days]

  def inferMove(self, n=1):
    self.ripl.infer('(gibbs move one %d)' % n)

  def inferPGibbs(self, p=10, n=1):
    for y in self.years:
      self.ripl.infer('(pgibbs %d ordered %d %d)' % (y, p, n))
  
  def inferHypers(self, n=1):
    self.ripl.infer('(mh hypers one %d)' % n)
  
  def sampleHypers(self):
    return [self.ripl.sample("(hypers %d)" % k) for k in range(num_features)]
  
  def likelihood(self):
    return self.ripl.get_global_logscore() - self.features_likelihood
