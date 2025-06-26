#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing module of the bnn4hi package

This module contains the main function to test the calibration of the
trained models generating the `reliability diagram`, test the accuracy
of the models with respect to the uncertainty of the predictions
generating the `uncertainty vs accuracy plot` and test the uncertainty
of each model, class by class, generating the `class uncertainty plot`
of each dataset.

This module can be imported as a part of the bnn4hi package, but it can
also be launched from command line, as a script. For that, use the `-h`
option to see the required arguments.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# Local imports
if '.' in __name__:

    # To run as a module
    from .lib import my_config
    from lib.data import get_dataset, get_filtered_dataset
    from .lib.bayesian_model import BayesianENet
    from .lib.analysis import *
    from .lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                           plot_accuracy_vs_uncertainty, plot_model_accuracy,
                           plot_confusion_matrix, plot_uncertainty_distribution)
else:

    # To run as a script
    from lib import my_config
    from lib.data import get_dataset, get_filtered_dataset, save_csv_uncertainty
    from lib.bayesian_model import BayesianENet
    from lib.analysis import *
    from lib.analysis import _predictive_entropy
    from lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                          plot_accuracy_vs_uncertainty, plot_model_accuracy,
                          plot_confusion_matrix, plot_uncertainty_distribution,plot_uncertainty_matrix)

# PARAMETERS
# =============================================================================

def _parse_args(dataset_list):
    """Analyses the received parameters and returns them organised.

    Takes the list of strings received at sys.argv and generates a
    namespace assigning them to objects.

    Parameters
    ----------
    dataset_list : list of str
        List with the abbreviated names of the datasets to test. If
        `test.py` is launched as a script, the received parameters must
        correspond to the order of this list.

    Returns
    -------
    out : namespace
        The namespace with the values of the received parameters
        assigned to objects.
    """

    # Generate the parameter analyser
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("data_path", default='/Data/hook/activations/', help="Ruta a la carpeta con las salidas de la capa anterior")
    parser.add_argument("csv_path", default='/data/train.csv', help="Ruta a la carpeta con los archivos CSV de datos")
    parser.add_argument("epochs", type=int, nargs=len(dataset_list), help=("List of the epoch of the selected checkpoint "
                              "for testing each model. The order must "
                              f"correspond to: {dataset_list}."))
    parser.add_argument("modelo", type=int, help="Número de modelo que se va a ejecutar")
    parser.add_argument('--train_data', type=int, default=0, help="Hacer test sobre los datos de entrenamiento")
    parser.add_argument('--my_models', type=int, default=1, help="Usar mis modelos")


    # Return the analysed parameters
    return parser.parse_args()

# PREDICT FUNCTIONS
# =============================================================================

def caracteristics(predictions, y_test, num_classes, samples=100, verbose=False,threshold= None, images_names=None):
    y_pred_mean = np.mean(predictions, axis=0).argmax(axis=1)

    if images_names is not None:
        #calculate_uncertainty_for_inputs(predictions, images_names, './Test/')
        calculate_uncertainty_with_labels(predictions, images_names, y_test, y_pred_mean,'./Test/')

    if verbose:
        print(f"Shape of predictions: {predictions.shape}")

    rd_data = (reliability_diagram(predictions, y_test,threshold=threshold))

    # Cross entropy and accuracy
    print("\nGenerating data for the `accuracy vs uncertainty` plot",
          flush=True)
    acc_data, px_data = accuracy_vs_uncertainty(predictions, y_test)

    print("\nGenerating data for the `class uncertainty` plot", flush=True)
    _, avg_Ep, avg_H_Ep = analyse_entropy(predictions, y_test)

    collect_distribution_all_classes(predictions, y_test, y_pred_mean)
    collect_distribution_without_uncertanty(y_test, y_pred_mean)

    uncertainty_data = collect_uncertainty_by_case(predictions, y_test, y_pred_mean, num_classes=num_classes)

    false_negatives_uncertainty = uncertainty_data["false_negatives"]
    false_positives_uncertainty = uncertainty_data["false_positives"]
    correct_predictions_uncertainty = uncertainty_data["correct_predictions"]
    uncertainty = _predictive_entropy(predictions)
    save_csv_uncertainty(images_names,y_test, uncertainty)

    return (rd_data, acc_data, px_data, avg_Ep, avg_H_Ep, y_pred_mean,
            false_negatives_uncertainty, false_positives_uncertainty, correct_predictions_uncertainty, uncertainty)


def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


def custom_accuracy(y_true, y_pred):
    y_pred_rounded = tf.math.round(y_pred)
    correct_predictions = tf.reduce_all(tf.equal(y_true, y_pred_rounded), axis=1)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

# MAIN FUNCTION
# =============================================================================

def test(epochs, verbose=False):
    """Tests the trained bayesian models

    The plots are saved in the `TEST_DIR` defined in `config.py`.

    Parameters
    ----------
    epochs : dict
        Dict structure with the epochs of the selected checkpoint for
        testing each model. The keys must correspond to the abbreviated
        name of the dataset of each trained model.
    """

    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------

    # Input, output and dataset references
    base_dir = my_config.MODELS_DIR
    datasets = my_config.DATASETS
    output_dir = my_config.TEST_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Model parameters
    l1_n = my_config.LAYER1_NEURONS
    l2_n = my_config.LAYER2_NEURONS

    # Training parameters
    p_train = my_config.P_TRAIN
    learning_rate = my_config.LEARNING_RATE

    # Bayesian passes
    passes = my_config.BAYESIAN_PASSES

    # Plot parameters
    colours = my_config.COLOURS
    w = my_config.PLOT_W
    h = my_config.PLOT_H

    # Plotting variables
    reliability_data = {}
    acc_data = {}
    px_data = {}

    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    for name, dataset in datasets.items():

        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']

        # Get model dir
        model_dir = (f"{name}_{l1_n}-{l2_n}model_{p_train}train"
                     f"_{learning_rate}lr")
        model_dir = os.path.join(base_dir, model_dir)

        print("############################################################")
        if not os.path.isdir(model_dir):
            reliability_data[name] = []
            acc_data[name] = []
            px_data[name] = []
            print("MODELO NO ENCONTRADO")
            exit()
        else:
            model_dir = os.path.join(model_dir, f"epoch_{epochs[name]}_{args.modelo}.pth")

        print("MODEL DIR: ", model_dir)
        # GET DATA
        # ---------------------------------------------------------------------
        # Get dataset
        
        csv.field_size_limit(sys.maxsize)
        df = pd.read_csv(args.csv_path)

        y =np.array(df['isup_grade'])
        image_names = np.array(df['image_id'])


        train_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        test_files = [os.path.join(args.data_path, f"feature_activations{n}.npy") for n in train_indices]
        activations = np.load(test_files[0])
        X_test = np.array(activations)
        y_test = y[:7724]
        y_test = np.array(y_test)


        # LOAD MODEL
        # ---------------------------------------------------------------------
        # Load trained model
        input_shape = X_test.shape[2:]
        print(X_test.shape)

        my_model = True if args.my_models == 1 else False
        if my_model:
            print(f"Modelo: {args.modelo}")
            model = BayesianENet(modelo=args.modelo, in_features=input_shape[0], output_dim=my_config.NUM_CLASES_TRAIN)
            model.summary(input_shape)
        else:
            print(f"Modelo original")
            model = nn.Linear(input_shape[0], my_config.NUM_CLASES_TRAIN)
        
        model.load_state_dict(torch.load(model_dir, weights_only=True))
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        device = next(model.parameters()).device
        model.eval()
        print("Modelo cargado")

        # model = tf.keras.models.load_model(model_dir)

        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------

        # Tests message
        print(f"\n### Starting {name} tests")
        print('#' * 80)
        print(f"\nMODEL DIR: {model_dir}")

        total_images = len(X_test)*3
        test_correct = 0
        y_pred = []
        X_test_tensor = []
       
        y_pred = []

        BATCH_SIZE =8

        print("Lanzando 100 predicciones bayesianas")
        with torch.no_grad():
            for i in range(50):
                status = int(78 * i / 50)
                print('[' + '=' * status + ' ' * (78 - status) + ']', end="\r", flush=True)
                y_pred_all_files = []

                for file in test_files:
                    activations = np.load(file, mmap_mode='r')
                    n_samples = activations.shape[0]

                    pred_batch = []

                    for start in range(0, n_samples, BATCH_SIZE):
                        end = min(start + BATCH_SIZE, n_samples)

                        X_batch = activations[start:end]
                        X_batch = np.squeeze(X_batch, axis=1) 

                        X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
                        prediction = model(X_tensor)
                        prediction = torch.sigmoid(prediction)
                        preds = [calculate_class(p) for p in prediction]
                        pred_batch.extend(preds)
                    y_pred_all_files.extend(pred_batch)
                y_pred.append(y_pred_all_files)
            y_pred = np.array(y_pred)

        # Launch predictions
        (reliability_data[name],
        acc_data[name],
        px_data[name],
        avg_Ep, avg_H_Ep, y_pred_mean,
        uncertainty_fn, uncertainty_fp, uncertainty_correct, uncertainty) = caracteristics(y_pred, y_test, num_classes, samples=passes, images_names=image_names)

        if verbose:
            print("y_test samples:", y_test[:10])
            print("y_pred samples:", y_pred_mean[:10])

            print("y_test shape:", y_test.shape)
            print("y_pred shape:", y_pred_mean.shape)

        # Generar la matriz de confusión
        classes = [f'Class {i}' for i in range(num_classes)]
        #plot_confusion_matrix(y_test, y_pred_mean, classes, output_dir, name, normalize=True)
        #plot_uncertainty_matrix(y_test, y_pred_mean, uncertainty, classes, output_dir, name)

        #plot_uncertainty_distribution(uncertainty_fn, output_dir, num_bins=10, uncertainty_type="Predictiva",
                                      #error_type= "falsos negativos")
        #plot_uncertainty_distribution(uncertainty_fp, output_dir, num_bins=10, uncertainty_type="Predictiva",
                                      #error_type="falsos positivos")
    
        # Liberate model
        del model

        # IMAGE-RELATED PLOTS
        # ---------------------------------------------------------------------
        # Plot class uncertainty
        #plot_class_uncertainty(output_dir, name, epochs[name], avg_Ep, avg_H_Ep, w, h)

    # End of tests message
    print("\n### Tests finished")
    print('#' * 80, flush=True)

    # Generate accuracy plot
    #plot_model_accuracy(acc_data, output_dir)

    # GROUPED PLOTS
    # -------------------------------------------------------------------------
    #plot_reliability_diagram(output_dir, reliability_data, w, h, colours)

    # Plot accuracy vs uncertainty
    #plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours)


if __name__ == "__main__":

    # Parse args
    dataset_list = my_config.DATASETS_LIST
    args = _parse_args(dataset_list)

    # Generate parameter structures for main function
    epochs = {}
    for i, name in enumerate(dataset_list):
        epochs[name] = args.epochs[i]

    # Launch main function
    test(epochs)

# TODO: TABLA DE ENTROPÍA SOBRE DATOS DE TRAIN (PARA TODOS LOS FALSOS NEGATIVOS)
# TODO: PREDICE     U        0   1   2   3   4   5
# TODO: 0        0-0.1      17% ..% ..% ..% ..% ..%   QUE SUME 100
# TODO: 0        0.1-0.2    ..% ..% ..% ..% ..% ..%  QUE SUME 100
# TODO: 0        0.2-0.3    ..% ..% ..% ..% ..% ..%  QUE SUME 100 
# ...
# TODO: 5        0.9-1.0    ..% ..% ..% ..% ..% ..%  QUE SUME 100
# TODO: FIJAR UMBRAL Y VOLVER A CONSTRUIR MATRIZ DE CONFUSIÓN, ELIGES LAS QUE EVALÚAS Y LAS QUE NO EVALÚAS SEGÚN EL UMBRAL



# TODO: FALSOS NEGATIVOS
# ENTRENAR CON CASOS SEGUROS (LOS QUE TENGAS MAS DE UN UNCERTANTY SE ELIMINAN DE TODO)
# TODO: QUÉ % NO CLASIFICARÍAMOS
