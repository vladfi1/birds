import venture.shortcuts as s
import numpy as np
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

def readReconstruction(dataset):
  path = "data/ground/dataset%d/" % dataset

  filename = None
  for f in os.listdir(path):
    if 'reconstruction' in f:
      filename = path + f
      break

  csv = loadCSV(filename)
  
  bird_moves = {}
  
  for row in csv[1:]:
    bird_moves[tuple(int(k)-1 for k in row[:4])] = float(row[4])
  
  return bird_moves

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

