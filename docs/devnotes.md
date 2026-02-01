## 20260131

1. Stepped through the REPA implementation for ChemProp. I think it works ok.
2. Got the posebusters and other validation metrics working as well.
3. Ran training runs mimicking the tabasco-mild configurationon dev gpu cluster. [Chemprop](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/tg4a8h91?nw=nwusersr2173) and [baseline](https://wandb.ai/sr2173-university-of-cambridge/tabasco/runs/oc3eb4x4?nw=nwusersr2173). The loss seems to reduce and converge faster on chemprop but I don't think the validation metrics seem especially better.
4. Still running into massive latency issues on HPC. Even in interactive mode, `import torch` seems to cause the process to hang. One hypothesis is that this is due to filesystem latency, but I'm not sure. Currently my code and venv is in /rds/... so maybe moving it to /home/... will help? TODO: investigate

## 20260130

Questions for investigation.
1. Implementation.
1.1. Is REPA for the last layer implemented correctly?
1.2. Are we using chemprop correctly?
1.3. Are we getting the final representation for the REPA encoder in the same way that the paper does it?
2. Why are the PoseBusters metrics not being captured right now?
3. Why are the training runs failing on the HPC?
4. Metrics
4.1. PoseBusters
4.2. Is convergence faster?

Step through this line by line!

## 20260121

1. Read through Tabasco paper and repo
2. Added v0 implementation of REPA addition
3. Ran inference successfully locally on some toy examples
3. Set up test dataset and test environment on local macbook-air

### Goals

Feb 1 — need to know if REPA works or if I have to change my project entirely
Feb 15
Mar 1
Mar 15
Apr 1
Apr 15
May 1
May 15 — experiments done
Jun 9 done

## 20251221

Reading about molecular generation, flow models, and such.
