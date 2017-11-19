import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def arg_parser():
    parser = argparse.ArgumentParser("Creates plots for data provided neural network")
    parser.add_argument('-d', '--datadir',
                        dest='data_dir',
                        type=str,
                        default="data-out",
                        help='Directory containing logs')
    parser.add_argument('-o', '--outputdir',
                        dest='output_dir',
                        type=str,
                        default=None,
                        help='Plots will be stored in output directory. '
                             'If none provided, output is stored in data directory')
    parser.add_argument('-ac', '--accuracy',
                        dest='accuracies',
                        type=str,
                        default="accuracies.csv",
                        help='Filename containing accuracies (default: accuracies.csv)')
    parser.add_argument('-du', '--durations',
                        dest='durations',
                        type=str,
                        default="durations.csv",
                        help='Filename containing durations (default: durations.csv)')
    return parser


def file_names(data_dir, regex):
    """
    Returns file name which match the given regex
    :param data_dir: directory to scan
    :return: list of file names
    """
    file_names = []
    for filename in os.listdir(data_dir):
        if re.match(regex, filename):
            file_names.append(os.path.join(data_dir, filename))
    return file_names


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir
    accuracies_file_name = args.accuracies
    durations_file_name = args.durations

    # Find all data files in data_dir
    regex = "^data.*\.txt$"  # matches all files beginning with 'data' and ending with '.txt'
    file_names = file_names(data_dir, regex)

    print("Creating Plots")
    print("Reading from: " + data_dir)
    print(" Data found")
    for file in file_names:
        print("   " + file)

    accuracy_file = Path(os.path.join(data_dir, accuracies_file_name))
    if accuracy_file.is_file():
        print("   " + accuracies_file_name)
    else:
        print("Accuracy file not found")

    durations_file = Path(os.path.join(data_dir, durations_file_name))
    if durations_file.is_file():
        print("   " + durations_file_name)
    else:
        print("Durations file not found")

    # --- Plot data
    plt.axis([0, 1000, 0.5, 1])
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")

    # We are only interested in the GPU data for this plot
    gpu_data = [file for file in file_names if str(file).endswith("GPU.txt")]
    for data in gpu_data:
        # Match learning rate in file name
        label = re.match("(.*)learnrate_(.*)-num", data).group(2)
        plt.plot(np.loadtxt(data), label=label)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracies-learning-rates.png"))
    plt.clf()

    # --- Plot durations
    gpu_points = []
    cpu_points = []
    with open(durations_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if "GPU" in row[2]:
                gpu_points.append((int(row[0]), float(row[1])))
            else:
                cpu_points.append((int(row[0]), float(row[1])))
    x_gpu = [x[0] for x in gpu_points]
    y_gpu = [x[1] for x in gpu_points]

    x_cpu = [x[0] for x in cpu_points]
    y_cpu = [x[1] for x in cpu_points]

    # scale to millions
    scale = 10e5
    x_gpu = [x / scale for x in x_gpu]
    x_cpu = [x / scale for x in x_cpu]

    plt.scatter(x_gpu, y_gpu, s=60, c='blue', marker='.', label="GPU")
    plt.scatter(x_cpu, y_cpu, s=60, c='green', marker='.', label="CPU")

    plt.xlim(0, 2.3)
    plt.xticks(x_gpu, rotation=75)  # This makes your desired x ticks
    plt.ylim(0, 10000)
    plt.title('Training duration in relation to number of parameters')
    plt.xlabel('Number of parameters in network [in millions]')
    plt.ylabel('Duration in seconds')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "durations-filter-size.png"))
    plt.clf()
