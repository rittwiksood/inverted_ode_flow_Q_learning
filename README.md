<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Flow Q-Learning</h1>
  
  </summary>
  </ul>
 <div>
 </div>

## Overview

Flow Q-learning (FQL) is a simple and performance data-driven RL algorithm
that leverages an expressive *flow-matching* policy
to model complex action distributions in data.

## Installation

To install the full dependencies, simply run:
```bash
pip install -r requirements.txt
```

## Usage

The main implementation of FQL is in [agents/fql.py](agents/fql.py),


## Tips for hyperparameter tuning

Here are some general tips for FQL's hyperparameter tuning for new tasks:

* The most important hyperparameter of FQL is the BC coefficient (`--agent.alpha`).
  This needs to be individually tuned for each environment.
* Although this was not used in the original paper,
  setting `--agent.normalize_q_loss=True` makes `alpha` invariant to the scale of the Q-values.
  **For new environments, we highly recommend turning on this flag** (`--agent.normalize_q_loss=True`)
  and tuning `alpha` starting from `[0.03, 0.1, 0.3, 1, 3, 10]`.
* For other hyperparameters, you may use the default values in `agents/fql.py`.
  For some tasks, setting `--agent.q_agg=min` (to enable clipped double Q-learning) may slightly improve performance.
  See the ablation study in the paper for more details.
* For pixel-based environments, don't forget to set `--agent.encoder=impala_small` (or larger encoders),
  `--p_aug=0.5`, and `--frame_stack=3`.

## Reproducing the main results
