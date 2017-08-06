import os

ensemble = ['frcnn', 'inception', 'rfcn']
batch = 10

print "First stage:"
for model in ensemble:
  print ">>", model
  os.system(' '.join(['python run_first_stage.py --model', model, '--batch', str(batch)]))

print "Ensemble of proposals"
os.system(' '.join(['python run_ensemble_first_stage.py --ensemble', '"' + ' '.join(ensemble) + '"']))

print "Second stage:"
for model in ensemble:
  print ">>", model
  os.system(' '.join(['python run_second_stage.py --model', model, '--batch', str(batch)]))

print "Ensemble of detections"
os.system(' '.join(['python run_ensemble_second_stage.py --ensemble', '"' + ' '.join(ensemble) + '"']))

print "Successfully done"

