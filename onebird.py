import venture.shortcuts as s

ripl = s.make_puma_church_prime_ripl()

from utils import *

# model
width = 4
height = 4
num_features = 4
total_birds = 1

ripl.assume('width', width)
ripl.assume('height', height)
ripl.assume('cells', '(* width height)')
ripl.assume('dist_max2', '18')

ripl.assume('cell2X', '(lambda (cell) (int_div cell height))')
ripl.assume('cell2Y', '(lambda (cell) (int_mod cell height))')
#ripl.assume('cell2P', '(lambda (cell) (make_pair (cell2X cell) (cell2Y cell)))')
ripl.assume('XY2cell', '(lambda (x y) (+ (* height x) y))')

ripl.assume('square', '(lambda (x) (* x x))')

ripl.assume('dist2', """
  (lambda (x1 y1 x2 y2)
    (+ 
      (square (- x1 x2))
      (square (- y1 y2))
    )
  )
""")

ripl.assume('cell_dist2', """
  (lambda (i j)
    (dist2
      (cell2X i) (cell2Y i)
      (cell2X j) (cell2Y j)
    )
  )
""")

ripl.assume('num_features', num_features)

# we want to infer the hyperparameters of a log-linear model
ripl.assume('hypers', """
  (mem (lambda (k)
    (scope_include
      (quote hypers) k
      (normal 0 1)
    )
  ))
""")

# the features will all be observed
ripl.assume('features', '(mem (lambda (y d i j k) (normal 0 1)))')

ripl.assume('fold', """
  (lambda (op f len)
    (if (= len 1) (f 0)
      (let ((l (- len 1)))
        (op (f l) (fold op f l))
      )
    )
  )
""")

# phi is the unnormalized probability of a bird moving
# from cell i to cell j between on day d
ripl.assume('phi', """
  (mem (lambda (y d i j)
    (if (> (cell_dist2 i j) dist_max2) 0
      (exp (fold +
        (lambda (k)
          (* (hypers k) (features y d i j k))
        )
        num_features
      ))
    )
  ))
""")

ripl.assume('array_from_func', """
  (lambda (f len)
    (if (= len 1) (array (f 0))
      (let ((l (- len 1)))
        (append
          (array_from_func f l)
          (f l)
        )
      )
    )
  )
""")

ripl.assume('get_bird_move_dist', """
  (mem (lambda (y d i)
      (to_simplex (array_from_func
        (lambda (j) (phi y d i j)) cells
      ))
  ))
""")

# samples where a bird would move to from cell i on day d
ripl.assume('move', """
  (lambda (y d i)
    (scope_include y d
      (categorical (get_bird_move_dist y d i))
    )
  )
""")

ripl.assume('get_bird_pos', """
  (mem (lambda (y d id)
    (if (= d 1) 0
      (move y (- d 1) (get_bird_pos y (- d 1) id))
    )
  ))
""")

ripl.assume('total_birds', total_birds)

ripl.assume('count_birds', """
  (lambda (y d i)
    (fold +
      (lambda (id)
        (if (= (get_bird_pos y d id) i) 1 0)
      )
      total_birds
    )
  )
""")

ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.01))))')

features_file = "release/onebird-features.csv"
observations_file = "release/onebird-observations.csv"

features = readFeatures(features_file)
observations = readObservations(observations_file)

def loadFeatures(years):
  for (y, d, i, j), fs in features.iteritems():
    if y not in years: continue
    
    for k, f in enumerate(fs):
      key = map(s.number, [y, d, i, j, k])
      ripl.sivm.observe(['features'] + key, s.number(f))

def loadObservations(years):
  for y in years:
    for (d, ns) in observations[y]:
      for i, n in enumerate(ns):
        print y, d, i
        ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)
      
      if d > 0:
        ripl.infer('(mh %d %d %d)' % (y, d-1, 10))

def getBirds(y, d):
  return [[ripl.sample('(count_birds %d %d (XY2cell %d %d))' % (y, d, x, y)) for y in range(height)] for x in range(width)]

def printYear(y):
  for d in range(1, 21):
    print getBirds(y, d)

def getBird(y, id):
  return [ripl.sample('(get_bird_pos %d %d %d)' % (y, d, id)) for d in range(1, 21)]



years = range(1, 2)

loadFeatures(years)
loadObservations(years)

