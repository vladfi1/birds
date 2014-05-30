import venture.shortcuts as s
import numpy as np
from scipy import misc
import os

def parseLine(line):
  return line.strip().split(',')

def loadCSV(filename):
  with open(filename) as f:
    return map(parseLine, f.readlines())

def update(dict, key, data):
  if key not in dict:
    dict[key] = []
  
  dict[key].append(data)

def readFeatures(filename):
  csv = loadCSV(filename)
  data = {}
  
  for row in csv[1:]:
    keys = tuple(int(k)-1 for k in row[:4])
    features = map(float, row[4:])
    data[keys] = features
  
  return data

def readObservations(filename):
  csv = loadCSV(filename)
  years = {}
  
  for row in csv[1:]:
    [year, day] = map(int, row[:2])
    cells = map(float, row[2:])
    update(years, year-1, (day-1, cells))
  
  return years

def readReconstruction(params):
  filename = "data/ground/dataset%d/10x10x%d-reconstruction-ground.csv" % (params["dataset"], params["total_birds"])
  csv = loadCSV(filename)
  
  bird_moves = {}
  
  for row in csv[1:]:
    bird_moves[tuple(int(k)-1 for k in row[:4])] = float(row[4])
  
  return bird_moves

def writeReconstruction(params, bird_moves):
  filename = "data/output/dataset%d/10x10x%d-reconstruction-ground.csv" % (params["dataset"], params["total_birds"])
  
  with open(filename, 'w') as f:
    for key, value in sorted(bird_moves.items()):
      f.write(','.join(map(str, [k+1 for k  in key] + [value])))
      f.write('\n')

def ensure(path):
  if not os.path.exists(path):
    os.makedirs(path)

def drawBirds(bird_locs, filename, width=None, height=None, **kwargs):
  scale = 10
  
  bitmap = np.ndarray(shape=(width*scale, height*scale))
  
  #import pdb; pdb.set_trace()
  
  for x in range(width):
    for xs in range(scale):
      for y in range(height):
        for ys in range(scale):
          bitmap[x*scale+xs, y*scale+ys] = bird_locs[x * height + y]
  
  print "Saving images to %s" % filename
  #plt.imshow(bitmap)
  misc.imsave(filename, bitmap)

def drawBirdMoves(bird_moves, path, cells=None, total_birds=None, years=None, days=None, **params):
  for y in years:
    bird_locs = [0] * cells
    bird_locs[0] = total_birds
    
    for d in days[:-1]:
      for i in range(cells):
        for j in range(cells):
          move = bird_moves[(y, d, i, j)]
          bird_locs[i] -= move
          bird_locs[j] += move
      
      p = path + '/%d' % y
      ensure(p)
      filename = p + '/%02d.png' % (d+1)
      drawBirds(bird_locs, filename, **params)

def drawGroundMoves(dataset):
  params = getParams(dataset)
  drawBirdMoves(readReconstruction(params), 'ground%d' % dataset, **params)

def getParams(dataset):
  params = {'dataset': dataset}

  if dataset == 1:
    params['width'] = params['height'] = 4
    params['total_birds'] = 1
    params['name'] = 'onebird'
    params['years'] = range(30)
    params['days'] = range(20)
  else:
    params['width'] = params['height'] = 10
    params['total_birds'] = 1000 if dataset == 2 else 1000000
    params['name'] = "%dx%dx%d-train" % (params['width'], params['height'], params['total_birds'])
    params['years'] = range(3)
    params['days'] = range(20)
    
  params['cells'] = params['width'] * params['height']

  return params

def toVenture(thing):
  if isinstance(thing, dict):
    return s.val("dict", {k:toVenture(v) for k, v in thing.iteritems()})
  if isinstance(thing, (list, tuple)):
    return s.val("array", [toVenture(v) for v in thing])
  if isinstance(thing, (int, float)):
    return s.number(thing)
  if isinstance(thing, str):
    return s.symbol(thing)

# handles numbers, lists, tuples, and dicts
def toExpr(thing):
  if isinstance(thing, dict):
    return dictToExpr(thing)
  if isinstance(thing, (list, tuple)):
    return listToExpr(thing)
  return str(thing)  

def expr(*things):
  return "(" + " ".join(map(toExpr, things)) + ")"

def dictToExpr(dict):
  return expr("dict", dict.keys(), dict.values())

def listToExpr(list):
  return expr("array", *list)

def fold(op, exp, counter, length):
  return '(' + op + " " + " ".join([exp.replace(counter, str(i)) for i in range(length)]) + ')'

def tree(op, exp, counter, lower, upper):
  average = (lower + upper) / 2
  if average == lower:
    return exp.replace(counter, str(lower))
  else:
    return '(' + op + " " + tree(op, exp, counter, lower, average) + ' ' + tree(op, exp, counter, average, upper) + ')'

from subprocess import call

def renderDot(dot,dirpath,i,fmt,colorIgnored):
  name = "dot%d" % i
  mkdir_cmd = "mkdir -p " + dirpath
  print mkdir_cmd
  call(mkdir_cmd,shell=True)
  dname = dirpath + "/" + name + ".dot"
  oname = dirpath + "/" + name + "." + fmt
  f = open(dname,"w")
  f.write(dot)
  f.close()
  cmd = ["dot", "-T" + fmt, dname, "-o", oname]
  print cmd
  call(cmd)

def renderRIPL(dirpath="graphs/onebird",fmt="svg",colorIgnored = False):
  dots = ripl.sivm.core_sivm.engine.getDistinguishedTrace().dot_trace(colorIgnored)
  i = 0
  for dot in dots:
    print "---dot---"
    renderDot(dot,dirpath,i,fmt,colorIgnored)
    i += 1

def avgFinalValue(history, name):
  series = history.nameToSeries[name]
  values = [s.values[-1] for s in series]
  return np.average(values)

def normalize(l):
  s = sum(l)
  return [e/s for e in l]
