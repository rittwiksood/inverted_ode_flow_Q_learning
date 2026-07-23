<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Flow Q-Learning</h1>
  
  </summary>
  </ul>
 <div>
 </div>

## Introduction

We have implemented an inverter to a ODE model. We use Flow Q-Learning to create this ODE. Flow Q-learning (FQL) is a data-driven RL algorithm that inhibits a flow-matching policy to model action distributions in data that are complex in nature. 


## Installation

1. conda env create -n <env_name> -f environment.yml

2. conda activate <env_name> 

## Usage

The main implementation of FQL is in [agents/fql.py],


## Reproducing the main results

1. Train the ODE model:

 We have trained on various environments with different tasks. This paper discusses results on the following environment:

 python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.q_agg=min --agent.alpha=10

