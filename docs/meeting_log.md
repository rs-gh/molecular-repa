## 20260122

### Workflow questions
1. Workflow for performing a training run? Where and how?
HPC, use 1 GPU. Miruna will send a script
2. What encoder should I start with? MACE? ChemProp?
Use chemprop to start, then move to MACE
Should we compute on the fly? -> maybe easy to do this for ChemProp so start with that
Or precompute? If using t=1 then we can precompute
3. What dataset?
Testing/setup: QM9, few 100k samples, up to 9 heavy atoms
For real results: use GEOM
4. What metrics?
PoseBusters validity
Overall validity

### Experiment questions
1. Does Tabasco still need positional encodings? Does the inductive bias from encoder alignment mean this is no longer necessary?
This could be interesting, run ablation study
2. What time steps is alignment most useful for? t ~ 0? t ~ 1?
Can start with t=1 and see if/how things change
3. Training curve for tabasco-mild? Should we compare against that?
TBD

Plan is to spend 1-2 weeks on small molecules, then move to protein models
