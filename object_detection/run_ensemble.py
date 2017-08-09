import os
import time

ensemble = ['frcnn', 'inception', 'rfcn']
batch = 100
nb_iter = 247


for i in range(nb_iter):
  start = time.time()
  print "Batch number: " + str(i+1)
  for model in ensemble:
    os.system(' '.join(['python run_first_stage.py --model', model, '--batch', str(batch)]))
  
  os.system(' '.join(['python run_ensemble_first_stage.py --ensemble', '"' + ' '.join(ensemble) + '"']))
  for model in ensemble:
    os.system(' '.join(['python run_second_stage.py --model', model, '--batch', str(batch)]))
  
  os.system(' '.join(['python run_ensemble_second_stage.py --ensemble', '"' + ' '.join(ensemble) + '"']))

  print "-> Done in " + str(time.time() - start)
  print 

print "Successfully done"

