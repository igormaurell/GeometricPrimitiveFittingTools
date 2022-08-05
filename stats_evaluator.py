import argparse

from tqdm import tqdm

import json

import numpy as np

from os import listdir
from os.path import join

import matplotlib.pyplot as plt

def createNestedPieGraph(labels, in_data, out_data, title='', number_in_per_out=1):
    fig, ax = plt.subplots()

    size = 0.6

    cmap = plt.get_cmap("tab20")
    out_color_array = np.arange(len(in_data))*2
    outer_colors = cmap(out_color_array)
    in_color_array = np.asarray([[x + i for i in range(0, number_in_per_out)]for x in out_color_array]).flatten()
    inner_colors = cmap(in_color_array)

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges = ax.pie(out_data, radius=1.3, colors=outer_colors, autopct=lambda pct: func(pct, in_data), pctdistance=0.85,
        wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(in_data, radius=1.3-size, colors=inner_colors, autopct=lambda pct: func(pct, in_data), pctdistance=1.15-size,
        wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title=title)

    ax.legend(wedges[0], labels,
            title="Types",
            loc="lower left",
            bbox_to_anchor=(1, 0, 0.5, 1), fontsize='large')

    return fig

def createPieGraph(labels, data, title=''):
    fig, ax = plt.subplots()

    size = 0.6

    cmap = plt.get_cmap("tab20")
    colors_array = np.arange(len(data))*2
    colors = cmap(colors_array)

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges = ax.pie(data, radius=1.3, colors=colors, autopct=lambda pct: func(pct, data), pctdistance=0.85,
        wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title=title)
    
    ax.legend(wedges[0], labels,
            title="Types",
            loc="lower left",
            bbox_to_anchor=(1, 0, 0.5, 1), fontsize='large')
    return fig

def createBarGraph(columns, data, title='', y_label=''):
    colors = plt.cm.BuGn(np.linspace(0.5, 1, len(data)))
    fig = plt.figure()
    plt.bar(columns, data, width=0.4, color=colors)
    plt.ylabel(y_label)
    plt.title(title)
    return fig

def statsData2Graphs(data):
    ##curves
    columns_curves = []
    data_number = []
    data_vertices = []
    for tp, d in data['curves'].items():
        columns_curves.append(tp)
        data_number.append(d['number_curves'])
        data_vertices.append(d['number_vertices'])
    fig_number_curves = createNestedPieGraph(columns_curves, data_number, data_number, title='Nuber of Curves per Type')
    fig_number_vertices = createPieGraph(columns_curves, data_vertices,  title='Nuber of Vertices per Type of Curve')

    ##surfaces
    columns_surfaces = []
    data_number = []
    data_faces = [] 
    for tp, d in data['surfaces'].items():
        columns_surfaces.append(tp)
        data_number.append(d['number_surfaces'])
        data_faces.append(d['number_faces'])
    fig_number_surfaces = createNestedPieGraph(columns_surfaces, data_number, data_number, title='Nuber of Surfaces per Type')
    fig_number_faces = createPieGraph(columns_surfaces, data_faces,  title='Nuber of Faces per Type of Surface')

    plt.show()
    #return fig_number_curves

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a dataset from OBJ and YAML to HDF5')
    parser.add_argument('folder', type=str, help='dataset folder.')

    parser.add_argument('--stats_folder_name', type=str, default = 'stats', help='stats folder name.')
    
    args = vars(parser.parse_args())

    folder_name = args['folder']
    stats_folder_name = join(folder_name, args['stats_folder_name'])

    stats_files_name = listdir(stats_folder_name)

    for filename in tqdm(stats_files_name):
        filename = join(stats_folder_name, filename)
        name = filename[0:filename.rfind('.')]
        data = {}
        with open(filename, 'r') as f:
            data = json.load(f)
        graph = statsData2Graphs(data)
        exit()