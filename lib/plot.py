#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot module of the bnn4hi package

The functions of this module are used to generate plots using the
results of the analysed bayesian predictions.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from collections import defaultdict

# Local imports
from .HSI2RGB import HSI2RGB


def plot_confusion_matrix(y_true, y_pred, classes, output_dir, name, normalize=True):
    """
    Plots the confusion matrix and saves it as an image. Additionally, saves
    another confusion matrix with raw counts (non-normalized).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    classes : list of str
        List of class labels.
    output_dir : str
        Directory to save the plot.
    name : str
        Name for the plot file.
    normalize : bool, optional (default: True)
        Whether to normalize the confusion matrix.
    """

    # Ajustar las clases basadas en y_true y y_pred
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    adjusted_classes = [classes[int(i)] for i in unique_classes]

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Guardar la matriz de confusión con números reales (sin normalización)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=adjusted_classes)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"{name} - Confusion Matrix (Counts)")
    real_count_path = os.path.join(output_dir, f"{name}_confusion_matrix_counts.png")
    plt.savefig(real_count_path)
    print(f"Saved raw count confusion matrix at {real_count_path}")
    plt.close()

    # Normalizar la matriz de confusión si es necesario
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Graficar y guardar la matriz de confusión normalizada
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=adjusted_classes)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
    plt.title(f"{name} - Confusion Matrix (Normalized)")
    normalized_path = os.path.join(output_dir, f"{name}_confusion_matrix_normalized.png")
    plt.savefig(normalized_path)
    print(f"Saved normalized confusion matrix at {normalized_path}")
    plt.show()
    plt.close()

    accuracy = (y_true == y_pred).astype(float).mean() * 100
    print(f"Accuracy: {accuracy.item():.2f}%")


def plot_uncertainty_distribution(uncertainty_by_class, output_dir, num_bins=10, uncertainty_type="Predictiva",
                                  error_type="falsos negativos"):
    """
    Grafica la distribución de incertidumbre para cada clase basada en los datos recopilados de errores (falsos positivos o falsos negativos).

    Parameters
    ----------
    uncertainty_by_class : dict
        Diccionario con listas de incertidumbre para cada clase en los errores especificados (falsos positivos o falsos negativos).
    output_dir : str
        Directorio para guardar el archivo PDF.
    num_bins : int, opcional
        Número de intervalos para el histograma (default: 10).
    uncertainty_type : str, opcional
        Tipo de incertidumbre que se grafica (Predictiva, Aleatoria, Epistémica).
    error_type : str, opcional
        Tipo de error a graficar ("falsos positivos" o "falsos negativos").
    """
    # Nombre del archivo PDF basado en el tipo de error
    pdf_file_path = os.path.join(output_dir, f"{error_type}_uncertainty_distribution.pdf")

    # Graficar la distribución de incertidumbre para cada clase
    with PdfPages(pdf_file_path) as pdf:
        plt.figure()
        for label, uncertainties in uncertainty_by_class.items():
            if uncertainties:  # Solo graficar si hay datos
                counts, bin_edges = np.histogram(uncertainties, bins=num_bins, range=(0, 1))
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                plt.plot(bin_centers, counts, label=f"Clase {label}")

                plt.xlabel("Incertidumbre")
                plt.ylabel("Cantidad de casos")
                plt.title(f"Distribución de Incertidumbre {uncertainty_type} para {error_type.capitalize()}")
                plt.legend()

                pdf.savefig()  # Guarda la figura actual en el PDF
                plt.close()  # Cierra la figura para liberar memoria y evitar solapamientos

        # Graficar todas las clases en una sola figura
        plt.figure()
        for label, uncertainties in uncertainty_by_class.items():
            if uncertainties:  # Solo graficar si hay datos
                counts, bin_edges = np.histogram(uncertainties, bins=num_bins, range=(0, 1))
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                plt.plot(bin_centers, counts, label=f"Clase {label}")

        plt.xlabel("Incertidumbre")
        plt.ylabel("Cantidad de casos")
        plt.title(f"Distribución de Incertidumbre {uncertainty_type} para {error_type.capitalize()} (Todas las clases)")
        plt.legend()

        pdf.savefig()
        plt.close()

    print(f"Incertidumbre guardadas en: {pdf_file_path}")



# MAP FUNCTIONS
# =============================================================================

def _uncertainty_to_map(uncertainty, num_classes, slots=10, max_H=0):
    """Groups the uncertainty values received into uncertainty groups
    
    Parameters
    ----------
    uncertainty : ndarray
        Array with the uncertainty values.
    num_classes : int
        Number of classes of the dataset.
    slots : int, optional (default: 10)
        Number of groups to divide uncertainty map values.
    max_H : float, optional (default: 0)
        The max value of the range of uncertainty for the uncertainty
        map. The `0` value will use the logarithm of `num_classes` as
        it is the theoretical maximum value of the uncertainty.
    
    Returns
    -------
    u_map : ndarray
        List with the uncertainty group corresponding to each
        uncertainty value received.
    labels : list of strings
        List of the labels for plotting the `u_map` value groups.
    """
    
    # Actualise `max_H` in case of the default value
    if max_H == 0:
        max_H = math.log(num_classes)
    
    # Prepare output structures and ranges
    u_map = np.zeros(uncertainty.shape, dtype="int")
    ranges = np.linspace(0.0, max_H, num=slots+1)
    labels = [f"0.0-{ranges[1]:.2f}"]
    
    # Populate the output structures
    slot = 1
    start = ranges[1]
    for end in ranges[2:]:
        
        # Fill with the slot number and actualise labels
        u_map[(start <= uncertainty) & (uncertainty <= end)] = slot
        labels.append(f"{start:.2f}-{end:.2f}")
        
        # For next iteration
        start = end
        slot +=1
    
    return u_map, labels


def plot_model_accuracy(acc_data, output_dir):
    """
    Plots the accuracy of the model for each dataset.

    Parameters
    ----------
    acc_data : dict
        A dictionary containing the accuracy data for each dataset.
    output_dir : str
        Directory where the plot will be saved.
    """

    # Extract dataset names and corresponding accuracies
    datasets = list(acc_data.keys())

    # Ensure that accuracies are numerical values, not arrays or lists

    accuracies = []
    for dataset in datasets:
        if isinstance(acc_data[dataset], (list, np.ndarray)):
            accuracy = np.mean(acc_data[dataset])
        else:
            accuracy = float(acc_data[dataset])
        accuracies.append(accuracy)

    print("Accuracies:", accuracies)

    # Save accuracies to a text file
    accuracies_file_path = os.path.join(output_dir, 'accuracies.txt')
    with open(accuracies_file_path, 'w') as f:
        for dataset, accuracy in zip(datasets, accuracies):
            f.write(f"{dataset}: {accuracy}\n")

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, accuracies, color='skyblue')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy for Each Dataset')
    plt.ylim(0, 1)  # Assuming accuracy is a value between 0 and 1

    # Save the plot
    plot_path = os.path.join(output_dir, 'model_accuracy.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Accuracy plot saved to {plot_path}")


def _map_to_img(prediction, shape, colours, metric=None, th=0.0, bg=(0, 0, 0)):
    """Generates an RGB image from `prediction` and `colours`
    
    The prediction itself should represent the index of its
    correspondent colour.
    
    Parameters
    ----------
    prediction : array_like
        Array with the values to represent.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    colours : list of RGB tuples
        List of colours for the RGB image representation.
    metric : array_like, optional (Default: None)
        Array with the same length of `prediction` to determine a
        metric for plotting or not each `prediction` value according to
        a threshold.
    th : float, optional (Default: 0.0)
        Threshold value to compare with each `metric` value if defined.
    bg : RGB tuple, optional (Default: (0, 0, 0))
        Background colour used for the pixels not represented according
        to `metric`.
    
    Returns
    -------
    img : ndarray
        RGB image representation of `prediction` colouring each group
        according to `colours`.
    """
    
    # Generate RGB image shape
    img_shape = (shape[0], shape[1], 3)
    
    if metric is not None:
        
        # Coloured RGB image that only shows those values where metric
        # is lower to threshold
        return np.reshape([colours[int(p)] if m < th else bg
                           for p, m in zip(prediction, metric)], img_shape)
    else:
        
        # Coloured RGB image of the entire prediction
        return np.reshape([colours[int(p)] for p in prediction], img_shape)

# PLOT FUNCTIONS
# =============================================================================

def plot_reliability_diagram(output_dir, data, w, h, colours, num_groups=10):
    """Generates and saves the `reliability diagram` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `reliability_diagram.pdf`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    data : dict
        It contains the `reliability diagram` data of each dataset. The
        key must be the dataset name abbreviation.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB colour of each dataset.
        The key must be the dataset name abbreviation.
    num_groups : int, optional (default: 10)
        Number of groups to divide xticks labels.
    """
    
    # Generate x axis labels and data for the optimal calibration curve
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    center = (p_groups[1] - p_groups[0])/2
    optimal = (p_groups + center)[:-1]
    if num_groups <= 10:
        labels = [f"{p_groups[i]:.1f}-{p_groups[i + 1]:.1f}"
                  for i in range(num_groups)]
    else:
        labels = [f"{p_groups[i]:.2f}-{p_groups[i + 1]:.2f}"
                  for i in range(num_groups)]
    
    # Xticks
    xticks = np.arange(len(labels))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for img_name in colours.keys():
        ax.plot(xticks[:len(data[img_name])], data[img_name], label=img_name,
                color=colours[img_name])
    ax.plot(xticks, optimal, label="Optimal calibration", color='black',
            linestyle='dashed')
    
    # Axes labels
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # Rotate X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "reliability_diagram.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)

def plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours,
                                 H_limit=1.2, num_groups=12, verbose=0, threshold=None):
    """Generates and saves the `accuracy vs uncertainty` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `accuracy_vs_uncertainty.pdf`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    acc_data : dict
        It contains the `accuracy vs uncertainty` data of each dataset.
        The key must be the dataset name abbreviation.
    px_data : dict
        It contains, for each dataset, the percentage of pixels of each
        uncertainty group. The key must be the dataset name
        abbreviation.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB colour of each dataset.
        The key must be the dataset name abbreviation.
    H_limit : float, optional (default: 1.5)
        The max value of the range of uncertainty for the plot.
    num_groups : int, optional (default: 15)
        Number of groups to divide xticks labels.
    """
    # Labels
    H_groups = np.linspace(0.0, H_limit, num_groups + 1)
    labels = [f"{H_groups[i]:.2f}-{H_groups[i + 1]:.2f}"
              for i in range(num_groups)]
    
    # Xticks
    xticks = np.arange(len(labels))
    
    # Yticks
    yticks = np.arange(0, 1.1, 0.1)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for img_name in colours.keys():
        ax.plot(xticks[:len(acc_data[img_name])], acc_data[img_name],
                label=f"{img_name} acc.", color=colours[img_name], zorder=3)
        ax.bar(xticks[:len(px_data[img_name])], px_data[img_name],
               label=f"{img_name} px %", color=colours[img_name], alpha=0.18,
               zorder=2)
        ax.bar(xticks[:len(px_data[img_name])],
               [-0.007 for i in px_data[img_name]], bottom=px_data[img_name],
               color=colours[img_name], zorder=3)
    
    # Axes label
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Pixels % and accuracy")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0] - 0.5, xticks[-1] + 0.5))
    
    # Y axis minors
    ax.set_yticks(yticks, minor=True)
    
    # Rotate X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y', zorder=1)
    ax.grid(axis='y', which='minor', linestyle='dashed', zorder=1)
    
    # Get legend handles and labels
    lg_handles, lg_labels = plt.gca().get_legend_handles_labels()

    if verbose:
        # Imprimir la información de las leyendas para depuración
        print(f"lg_handles: {len(lg_handles)} handles")
        print(f"lg_labels: {len(lg_labels)} labels")

    # Manual legend to adjust the handles and place labels in a new order
    order = [i for i in range(min(len(lg_handles), len(lg_labels)))]

    ax.add_artist(ax.legend([lg_handles[idx] for idx in order],
                            [lg_labels[idx] for idx in order],
                            loc='upper center', ncol=5,
                            bbox_to_anchor=(0.46, 1.2)))

    # Manually added handles upper lines (to match the bars)
    if len(lg_handles) > 0:
        ax.add_artist(ax.legend([lg_handles[0]], [""], framealpha=0,
                                handlelength=1.8, loc='upper center',
                                bbox_to_anchor=(-0.0555, 1.146)))
    if len(lg_handles) > 1:
        ax.add_artist(ax.legend([lg_handles[1]], [""], framealpha=0,
                                handlelength=1.8, loc='upper center',
                                bbox_to_anchor=(0.177, 1.146)))
    if len(lg_handles) > 2:
        ax.add_artist(ax.legend([lg_handles[2]], [""], framealpha=0,
                                handlelength=1.8, loc='upper center',
                                bbox_to_anchor=(0.3947, 1.146)))
    if len(lg_handles) > 3:
        ax.add_artist(ax.legend([lg_handles[3]], [""], framealpha=0,
                                handlelength=1.8, loc='upper center',
                                bbox_to_anchor=(0.6406, 1.146)))
    if len(lg_handles) > 4:
        ax.add_artist(ax.legend([lg_handles[4]], [""], framealpha=0,
                                handlelength=1.8, loc='upper center',
                                bbox_to_anchor=(0.8698, 1.146)))
    
    # Save
    if threshold is None:
        file_name = "accuracy_vs_uncertainty.pdf"
    else:
        file_name =f"accuracy_vs_uncertainty_{threshold}.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)


def plot_class_uncertainty(output_dir, name, epoch, avg_Ep, avg_H_Ep, w, h,
                           colours=["#2B4162", "#FA9F42", "#0B6E4F"], verbose=0):
    """Generates and saves the `class uncertainty` plot of a dataset
    
    It saves the plot in `output_dir` in pdf format with the name
    `<NAME>_<EPOCH>_class_uncertainty.pdf`, where <NAME> is the
    abbreviation of the dataset name and <EPOCH> the number of trained
    epochs of the tested checkpoint.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    epoch : int
        The number of trained epochs of the tested checkpoint.
    avg_Ep : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    avg_H_Ep : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : list, optional
        (default: ["#2B4162", "#FA9F42", "#0B6E4F"])
        List with the str format of the HEX value of at least three RGB
        colours.
    """
    
    # Xticks
    xticks = np.arange(len(avg_Ep))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    ax.bar(xticks, avg_Ep, label="Ep", color=colours[0], zorder=3)
    ax.bar(xticks, avg_H_Ep, bottom=avg_Ep, label="H - Ep",
           color=colours[2], zorder=3)
    
    # Highlight avg border
    ax.bar(xticks[-1], avg_Ep[-1] + avg_H_Ep[-1], zorder=2,
           edgecolor=colours[1], linewidth=4)
    
    # Axes label
    ax.set_xlabel(f"{name} classes")
    
    # X axis labels
    ax.set_xticks(xticks)
    xlabels = np.append(xticks[:-1], ["AVG"])
    ax.set_xticklabels(xlabels)
    
    # Grid
    ax.grid(axis='y', zorder=1)
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = f"{name}_{epoch}_class_uncertainty.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)

def plot_maps(output_dir, name, shape, num_classes, wl, img, y, pred_map,
              H_map, colours, gradients, max_H=1.5, slots=15):
    """Generates and saves the `uncertainty map` plot of a dataset
    
    This plot shows an RGB representation of the hyperspectral image,
    the ground truth, the prediction map and the uncertainty map.
    
    It saves the plot in `output_dir` in pdf format with the name
    `H_<NAME>.pdf`, where <NAME> is the abbreviation of the dataset
    name.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    num_classes : int
        Number of classes of the dataset.
    wl : list of floats
        Selected wavelengths of the hyperspectral image for RGB
        representation.
    img : ndarray
        Flattened list of the hyperspectral image pixels normalised.
    y : ndarray
        Flattened ground truth pixels of the hyperspectral image.
    pred_map : ndarray
        Array with the averages of the bayesian predictions.
    H_map : ndarray
        Array with the global uncertainty (H) values.
    colours : list of RGB tuples
        List of colours for the prediction map classes.
    gradients : list of RGB tuples
        List of colours for the uncertainty map groups of values.
    max_H : float, optional (default: 1.5)
        The max value of the range of uncertainty for the uncertainty
        map.
    slots : int, optional (default: 15)
        Number of groups to divide uncertainty map values.
    """
    
    # PREPARE FIGURE
    # -------------------------------------------------------------------------
    
    # Select shape and size depending on the dataset
    if name in ["IP", "KSC"]:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(2*shape[1]/96, 2*shape[0]/96)
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.set_size_inches(4*shape[1]/96, shape[0]/96)
    
    # Remove axis
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    
    # RGB IMAGE GENERATION
    #     Using HSI2RGB algorithm from paper:
    #         M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O.
    #         Ulfarsson, H. Deborah and J. R. Sveinsson, "Creating RGB
    #         Images from Hyperspectral Images Using a Color Matching
    #         Function," IGARSS 2020 - 2020 IEEE International
    #         Geoscience and Remote Sensing Symposium, 2020,
    #         pp. 2045-2048, doi: 10.1109/IGARSS39084.2020.9323397.
    #     HSI2RGB code from:
    #         https://github.com/JakobSig/HSI2RGB
    # -------------------------------------------------------------------------
    
    # Create and show RGB image (D65 illuminant and 0.002 threshold)
    RGB_img = HSI2RGB(wl, img, shape[0], shape[1], 65, 0.002)
    ax1.imshow(RGB_img)
    
    # GROUND TRUTH GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured ground truth
    gt = _map_to_img(y, shape, [(0, 0, 0)] + colours[:num_classes])
    ax2.imshow(gt)
    
    # PREDICTION MAP GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured prediction map
    pred_H_img = _map_to_img(pred_map, shape, colours[:num_classes])
    ax3.imshow(pred_H_img)
    
    # UNCERTAINTY MAP GENERATION
    # -------------------------------------------------------------------------
    
    # Create uncertainty map
    u_map, labels = _uncertainty_to_map(H_map, num_classes, slots=slots,
                                        max_H=max_H)
    
    # Generate and show coloured uncertainty map
    H_img = _map_to_img(u_map, shape, gradients[:slots])
    ax4.imshow(H_img)
    
    # PLOT COMBINED IMAGE
    # -------------------------------------------------------------------------
    
    # Adjust layout between images
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    
    # Save
    file_name = f"H_{name}.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)

def plot_uncertainty_with_noise(output_dir, name, labels, data, w, h,
                                colours=["#2B4162", "#FA9F42", "#0B6E4F"]):
    """Generates and saves the `noise` plot of a dataset
    
    It saves the plot in `output_dir` in pdf format with the name
    `<NAME>_noise.pdf`, where <NAME> is the abbreviation of the dataset
    name.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    labels : ndarray
        List of the evaluated noises that will be used as xlabels.
    data : ndarray
        It contains the class predictions for the list of noises of
        the dataset. The two last positions correspond to the average
        and the maximum uncertainty value prepared to be plotted.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : list, optional
        (default: ["#2B4162", "#FA9F42", "#0B6E4F"])
        List with the str format of the HEX value of at least three RGB
        colours.
    """
    
    # Add the data for plotting the maximum uncertainty line
    max_uncertainty = np.log(len(data) - 1)
    data = np.concatenate((data, [[max_uncertainty] * len(data[0])]))
    
    # Labels and xticks
    xticks = np.linspace(0.0, labels[-1], 13)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    #     Some of the colours of other plots are used here o it is no
    #     necessary to define different colours for this plot
    for n, d in enumerate(data[:-3]):
        ax.plot(labels, d, color=colours[0])
    ax.plot(labels, data[-3], color=colours[0], label="classes")
    ax.plot(labels, data[-2], color=colours[1], label="avg",
            linestyle='dashed')
    ax.plot(labels, data[-1], color=colours[2], label="max",
            linestyle='dashed')
    
    # Axes label
    ax.set_xlabel("Noise factor")
    ax.set_ylabel("Uncertainty")
    
    # Y axis limit
    y_lim = np.ceil(max_uncertainty)
    if max_uncertainty <= y_lim - 0.5:
        y_lim -= 0.5
    ax.set_ylim((0, y_lim))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(xticks, 2))
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = f"{name}_noise.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)


def plot_combined_noise(output_dir, labels, data, w, h, colours):
    """Generates and saves the `combined noise` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `combined_noise.pdf`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    labels : ndarray
        List of the evaluated noises that will be used as xlabels.
    data : dict
        It contains the normalised average predictions for the list of
        noises of each dataset. The key must be the dataset name
        abbreviation.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB colour of each dataset.
        The key must be the dataset name abbreviation.
    """
    
    # Labels and xticks
    xticks = np.linspace(0.0, labels[-1], 11)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for name, d in data.items():
        ax.plot(labels, d, color=colours[name], label=name)
    
    # Axes label
    ax.set_xlabel("Noise factor")
    ax.set_ylabel("Uncertainty")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(xticks, 2))
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "combined_noise.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)


def plot_mixed_uncertainty(output_dir, name, epoch, data, class_a, class_b, w,
                           h, colours=["#2B4162", "#D496A7"]):
    """Generates and saves the `mixed classes` plot of a dataset
    
    It saves the plot in `output_dir` in pdf format with the name
    `<NAME>_<EPOCH>_mixed_classes.pdf`, where <NAME> is the
    abbreviation of the dataset name and <EPOCH> the number of trained
    epochs of the tested checkpoint.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    epoch : int
        The number of trained epochs of the tested checkpoint.
    data : list of lists
        Contains one list for the normal model predictions with the
        aleatoric uncertainty (Ep) values of the mixed classes and the
        dataset average, and other list with the same information for
        the mixed model predictions.
    class_a : int
        Number of the first mixed class.
    class_b : int
        Number of the second mixed class.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : list, optional
        (default: ["#2B4162", "#D496A7"])
        List with the str format of the HEX value of at least two RGB
        colours.
    """
    
    # Xticks
    xticks = np.arange(len(data[0]))
    xticks_0 = xticks - 0.21
    xticks_1 = xticks + 0.21
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    ax.bar(xticks_0, data[0], label="Ep", color=colours[0], width=0.35,
           zorder=3)
    ax.bar(xticks_1, data[1], label="Ep mixed", color=colours[1],
           width=0.35, zorder=3)
    
    # Axes label
    ax.set_xlabel(f"{name} mixed classes")
    
    # X axis labels
    ax.set_xticks(xticks)
    xlabels = [f"class {class_a}", f"class {class_b}", "avg. (all classes)"]
    ax.set_xticklabels(xlabels)
    
    # Grid
    ax.grid(axis='y', zorder=1)
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = f"{name}_{epoch}_mixed_classes.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print(f"Saved {file_name} in {output_dir}", flush=True)


def plot_uncertainty_matrix(y_true, y_pred, uncertainties, classes, output_dir, name):
    """
    Plots and saves a matrix showing the average uncertainty per (true, predicted) class pair.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    uncertainties : array-like of shape (n_samples,)
        Uncertainty value for each prediction.
    classes : list of str
        List of class names.
    output_dir : str
        Path to save the output figure.
    name : str
        Filename prefix.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    uncertainties = np.array(uncertainties)

    num_classes = len(classes)
    unc_dict = defaultdict(list)

    # Recolectar incertidumbres por celda (true, pred)
    for t, p, u in zip(y_true, y_pred, uncertainties):
        unc_dict[(t, p)].append(u)

    # Construir la matriz de incertidumbre promedio
    unc_matrix = np.full((num_classes, num_classes), np.nan)  # NaN si no hay ejemplos

    for i in range(num_classes):
        for j in range(num_classes):
            values = unc_dict.get((i, j), [])
            if values:
                unc_matrix[i, j] = np.mean(values)

    # Crear el heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        unc_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=plt.cm.Blues, 
        xticklabels=classes, 
        yticklabels=classes, 
        cbar_kws={'label': 'Incertidumbre Promedio'}
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta Verdadera")
    plt.title(f"{name} - Matriz de Incertidumbre Promedio")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_uncertainty_matrix.png")
    plt.savefig(out_path)
    print(f"Saved uncertainty matrix at {out_path}")
    plt.close()

def plot_threshold_uncertainty (preds, labels, uncertainty, num_classes, output_path):
    umbrales = [0.0075, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    results = []

    total_samples = len(labels)

    for threshold in umbrales:
        mask = uncertainty < threshold
        
        fn = np.sum((labels[mask] == 1) & (preds[mask] == 0))
        
        if np.sum(mask) > 0:
            acc = np.mean(preds[mask] == labels[mask]) * 100
        else:
            acc = np.nan

        percent = np.sum(mask) / total_samples * 100

        results.append([
            threshold, fn, acc, percent
        ])

    file_name = "analisis_umbral.csv"
    df = pd.DataFrame(results, columns=["Umbral", "Falsos Negativos", "Accuracy (%)", "% Casos dentro del Umbral"])
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(f"Resultados guardados en: {output_path}")
