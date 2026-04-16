[Link](https://docs.google.com/document/d/1w3ZjzOyenn6WR1nb8BlDr7PeQuWhVy2kOcmvbYJfIXc/edit?tab=t.0) to running notes.

## 20260129

### Updates
- Set up REPA infra with chemprop for Tabasco, though currently this only aligns with the final transformer layer. (This was just the quickest way to get something up and running.) TODOs:
 - Setup REPA to work with intermediate layers. This needs some medium-lift code tweaking to extract and elevate state.
 - Code review? Have I done this right? Currently using Chemeleon weights and computing representations on the fly.
 - Set up MACE encoder, following Rokas’s code. Started taking a look. Will report soon.
- HPC credentials set up, ran some slurm batch jobs to test on 1 GPU.
- Reading/learning about Proteina. Hoping to do something with this for the L65 project as well.

### Questions
REPA/Tabasco: Which layers should I align with?
 - The final layer is good for now. Get a v0 working here, make this the priority. Intermediate layers could be an ablation. Get a basic codebase up to get metrics
 - Unintuitive to predict what layer is most representative, this will be quite empirical
 - Consider using SDPA attention?

Metrics: What metrics should we be measuring? Having some trouble getting PoseBusters running at least locally. Not sure why.
 - Qm9 PB with and without REPA last layer
 - Eventually use GEOM-drugs: see how this affects PB metrics
 - Is convergence faster?

Data: Currently using qm9. Is this sufficient?
 - Yes for now

Encoder computation: I am currently using y from the train set to get an encoder representation (i.e. t=1). Is this ok for now?
 - Just do what REPA does
 - Should you predict x1 or use the train set x1? Match what the paper does

Comparison: What training curve should we compare results with? Tabasco-mild?

HPC Debugging: Is there a good workflow to debug slurm jobs?
 - Hydra lightning debug mode – use 1 batch to check if fwd/backward passes work + if validation passes work
 - Check out https://github.com/xl0/lovely-tensors
 - Does the flow matching time look ok? Ensure that you get good coverage across 0 -> 1
 - Check timing for forward pass
 - Look at a single training step and debug that – working things out on a single batch and overfit that

### TODOs for next week

Project
Schedule: I need to break this down more granularly, and will send out something for feedback.


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

Goal for 20260129: Run Tabasco with and without REPA using chemprop, see what we get
