# Directories
- report
    - contains the report for this exercise
 - data
    - contains MNIST data set
- data-out
    - contains output of the training runs, such as learning curves, tensorflow summaries etc.
    - logs
        - contains tensorflow summaries generated during traing
    - data-* files contains same data as tensorflow summaries, used for plot generation of the first exercise
    - accuracies.csv and accuracies_final.csv:
        - contains accuracies of the different runs on the test set
        _ accuracies_final contains the values relevant for the first experiment of the exercise
        - images with prefix accuracies_ are generated from the data-* files and display the learning curve
    - durations.csv and durations_final.csv:
        - contains durations of the different runs
        - durations_final contains the values relevant for the second experiment of the exercise.



# Example Usage
## Run (train network)
```
python cnn.py -dd data-out -ld data-out/logs -e 1000 -l 0.1
```
with learning rate = 0.1, data stored in data-out and storing the tensorflow log to data-out/logs.

Run
```
python cnn.py -h
```
for help.


## Create Plots

Run
```
python plot.py
```
to generate plots from data-out directory. See -h option for parameters.


# TensorBoard
It is possible to view the results of the runs on TensorBoard.
By default, the TensorFlow output is stored in /tmp, see command line arguments for changing the output directories.

The runs for the exercise sheet are stored in data-out.
The data-out directory consists of several data files and a TensorFlow log directory with the different runs stored.

To inspect the runs in TensorBoard, run the following command:
```
cd <exercise-02-directory>
tensorboard --logdir=exercise-02/data-out/logs
```

# Use TensorFlow in PyCharm

For TensorFlow to work with the Run/Debug function of PyCharm, we need to add the CUDA environment variables to the Run / Debug Configuration.

In IntelliJ / PyCharm:
1. Edit Configuration
2. Open your run / debug configuration
3. Add following environment variables

```
LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:/opt/cuda/lib64"
CUDA_HOME = /opt/cuda/
```

The variables should match the ones in the .bashrc (or .zshrc etc).

# Run on CPU
There are two ways of running TensorFlow on the CPU. The first can be used in code.

The second method sets an environment variable to a certain value. This approach can be used with any tensorflow program and works without chaning the implementation.


