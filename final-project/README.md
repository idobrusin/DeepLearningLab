# DQN variants
This project implements DQN, DQN with target network and Dueling DQN in order to compare the performance in a grid maze world.

# Results 
DQN, target DQN and DDQN were trained using the default network provided in the code, Dueling DQN was trained using the config file cli (See below for run instructions)

The data/logs folder contains several training and test runs, use tensorboard with  `--logdir=data/logs` option to see results. 

The folder data/logs/HyperOpt contains several models for Duelling DQN which were trained with the config files located in config folder. 

# Usage
You can run the run_agent.py script in order to train and test an agent.
It is either possible to use the default networks for the different algorithms or by providing a .cfg file containing the hyperparameters to be used.\

## Minimal Example
```
# --------- Use default networks ---------
# run_name = name of run (log file name etc.)
# TRAIN = train or test mode
# 1 = DQN, 2 = Target network, 3 = DDQN, 4 = Dueling DQN

# Train agent  ---------
python run_agent.py cli run_name TRAIN 1 
# Test agent ---------
python run_agent.py cli run_name TEST 1 

# -----------------------------------
# --------- Use Config file --------- 
# config_file_name has to be a .cfg file in the config directory
python run_agent.py cfg config_file_name TRAIN 

```

## General
```
# Usage
usage: run_agent.py [-h] [-V] [--plot_output] [-G] {cfg,cli} ... {TRAIN,TEST}

positional arguments:
  {TRAIN,TEST}     Type of run, available values: TRAIN - train an agent, TEST
                   - test an agent.

optional arguments:
  -h, --help       show this help message and exit
  -V, --visualize  Visualize simulation.
  --plot_output    Enable plotting of the final results
  -G, --graph      Debugging option. Terminates run after generating graph.

Mode of Operation:
  {cfg,cli}        Choose how the model and simulation parameters are
                   provided.
    cfg            Subcommand to use a configuration file.
    cli            Subcommand to read configuration from the CLI.


# Train agent with given configuration (without file extension .cfg)
python run_agent.py cfg <config> TRAIN

```



# References
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df

https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/

https://arxiv.org/pdf/1511.05952.pdf
