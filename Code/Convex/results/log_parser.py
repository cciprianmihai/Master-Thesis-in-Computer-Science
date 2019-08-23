import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

rx_dict = {
    'pytorch': re.compile(r'Pytorch: (?P<pytorch>.*)\n'),
    'optimizer': re.compile(r'Optimizer: (?P<optimizer>.*)\n'),
    'epoch': re.compile(r'Epoch: \[(?P<epoch>\d+)\]'),
    'test': re.compile(r'Test accuracy: (?P<test>.*?)_'),
    'loss': re.compile(r'Loss: (?P<loss>.*?)_'),
    'accuracy': re.compile(r'Accuracy: (?P<accuracy>.*?)_'),
    'lr': re.compile(r'Learning rate: (?P<lr>.*)\n'),
    'validation': re.compile(r'Validation accuracy: (?P<validation>.*?)_'),
    'batch_size': re.compile(r'batch_size=(?P<batch_size>.*?),'),
    'progress': re.compile(r'Progress: (?P<progress>.*?)\n')
}

def _parse_line(line):
    res = []
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            res.append([key, match])
    if res == []:
        return None
    else:
        return res

def parse_file(filepath):
    epochs = []
    losses = []
    accuracies_train = []
    accuracies_validation = []
    accuracies_test = []
    progresses = []
    lrs = []
    optimizer = ''
    batch_size = 0
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            parse = _parse_line(line)
            if parse is not None:
                for elem in parse:
                    key, val = elem
                    if(key == 'epoch'):
                        epochs.append(int(val.group(key)))
                    elif(key == 'loss'):
                        losses.append(float(val.group(key)))
                    elif(key == 'accuracy'):
                        accuracies_train.append(float(val.group(key)))
                    elif(key == 'test'):
                        accuracies_test.append(float(val.group(key)))
                    elif (key == 'validation'):
                        accuracies_validation.append(float(val.group(key)))
                    elif(key == 'optimizer'):
                        optimizer = val.group(key)
                    elif(key == 'lr'):
                        lrs.append(float(val.group(key)))
                    elif (key == 'batch_size'):
                        batch_size = val.group(key)
                    elif (key == 'progress'):
                        progresses.append(float(val.group(key)))
            line = file_object.readline()

    return epochs, losses, accuracies_train, accuracies_test, accuracies_validation, optimizer, lrs, batch_size, progresses

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

# if batched = True, append to the optimizer name, the batch size
def make_plots(plot_name, batches=False, directory='logs'):
    accuracies_train_plot_name = plot_name + '_train.png'
    losses_plot_name = plot_name + '_loss.png'
    accuracies_validation_plot_name = plot_name + '_validation.png'
    accuracies_test_plot_name = plot_name + '_test.png'
    lr_plot_name = plot_name + '_lr.png'
    progresses_plot_name = plot_name + '_progress.png'
    colors = ['red', 'blue', 'green', 'orange', 'violet', 'cyan', 'yellow', 'magenta']
    files_list = files(directory)
    list_epochs = []
    list_accuracies_train = []
    list_losses = []
    list_accuracies_test = []
    list_accuracies_validation = []
    list_optimizers = []
    list_lrs = []
    list_progresses = []
    for file_list in files_list:
        epochs, losses, accuracies_train, accuracies_test, accuracies_validation, optimizer, lrs, batch_size, progresses = parse_file(directory + '/' + file_list)
        list_epochs.append(epochs)
        list_losses.append(losses)
        list_accuracies_train.append(accuracies_train)
        list_accuracies_test.append(accuracies_test)
        list_accuracies_validation.append(accuracies_validation)
        if batches == True:
            optimizer = optimizer + "_" + str(batch_size)
        list_optimizers.append(optimizer)
        list_lrs.append(lrs)
        list_progresses.append(progresses)

    # Plot accuracies train
    for i, accuracies_train in enumerate(list_accuracies_train):
        df_accuracies_train = pd.DataFrame(dict(epochs=list_epochs[i], accuracies=accuracies_train))
        sns.lineplot('epochs', 'accuracies', data=df_accuracies_train, color=colors[i], label=list_optimizers[i])
    plt.savefig(accuracies_train_plot_name)
    plt.show()

    # Plot losses
    for i, losses in enumerate(list_losses):
        df_losses = pd.DataFrame(dict(epochs=list_epochs[i], losses=losses))
        sns.lineplot('epochs', 'losses', data=df_losses, color=colors[i], label=list_optimizers[i])
    plt.savefig(losses_plot_name)
    plt.show()

    # Plot accuracies test
    for i, accuracies_test in enumerate(list_accuracies_test):
        df_accuracies_test = pd.DataFrame(dict(epochs=range(len(accuracies_test)), accuracies=accuracies_test))
        sns.lineplot('epochs', 'accuracies', data=df_accuracies_test, color=colors[i], label=list_optimizers[i])
    plt.savefig(accuracies_test_plot_name)
    plt.show()

    # Plot accuracies validation
    for i, accuracies_validation in enumerate(list_accuracies_validation):
        df_accuracies_validation = pd.DataFrame(dict(epochs=range(len(accuracies_validation)), accuracies=accuracies_validation))
        sns.lineplot('epochs', 'accuracies', data=df_accuracies_validation, color=colors[i], label=list_optimizers[i])
    plt.savefig(accuracies_validation_plot_name)
    plt.show()

    # Plot learning rates
    for i, lr in enumerate(list_lrs):
        df_lrs = pd.DataFrame(dict(evolution=range(len(lr)), lrs=lr))
        sns.lineplot('evolution', 'lrs', data=df_lrs, color=colors[i], label=list_optimizers[i])
    plt.savefig(lr_plot_name)
    plt.show()

    # Plot progresses
    for i, progress in enumerate(list_progresses):
        df_progresses = pd.DataFrame(dict(time=range(len(progress)), progress=progress))
        sns.lineplot('time', 'progress', data=df_progresses, color=colors[i], label=list_optimizers[i])
    plt.savefig(progresses_plot_name)
    plt.show()

make_plots('rn50_sgd', False)
