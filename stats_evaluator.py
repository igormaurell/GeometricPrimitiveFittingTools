import argparse

from tqdm import tqdm

import json

import numpy as np

from shutil import rmtree
from os import listdir, makedirs
from os.path import join

import matplotlib.pyplot as plt

def addStats(dict1, dict2):
    for key, value in dict2.items():
        if type(value) == int:
            if key in dict1:
                dict1[key] += value
            else:
                dict1[key] = value
        if type(value) == dict:
            if key in dict1:
                dict1[key] = addStats(dict1[key], value)
            else:
                dict1[key] = addStats({}, value)
    return dict1

def createNestedPieGraph(labels, in_data, out_data, title='', number_in_per_out=1):
    fig, ax = plt.subplots()

    size = 0.8

    cmap = plt.get_cmap("tab20")
    out_color_array = np.arange(len(in_data))*2
    outer_colors = cmap(out_color_array)
    in_color_array = np.asarray([[x + i for i in range(0, number_in_per_out)]for x in out_color_array]).flatten()
    inner_colors = cmap(in_color_array)

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges = ax.pie(out_data, radius=1.1, colors=outer_colors, autopct=lambda pct: func(pct, in_data), pctdistance=0.85,
        wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(in_data, radius=1.1-size, colors=inner_colors, autopct=lambda pct: func(pct, in_data), pctdistance=1.15-size,
        wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title=title, loc='left')

    ax.legend(wedges[0], labels,
            title="Types",
            loc="lower right",
            bbox_to_anchor=(0.6, 0, 0.5, 1), fontsize='large')

    return fig

def createPieGraph(labels, data, title=''):
    fig, ax = plt.subplots()

    size = 0.8

    cmap = plt.get_cmap("tab20")
    colors_array = np.arange(len(data))*2
    colors = cmap(colors_array)

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges = ax.pie(data, radius=1.1, colors=colors, autopct=lambda pct: func(pct, data), pctdistance=0.65,
        wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title=title)
    
    ax.legend(wedges[0], labels,
            title="Types",
            loc="lower right",
            bbox_to_anchor=(0.85, -0.1, 0.5, 1), fontsize='large')
    return fig

def createBarGraph(columns, data, title='', y_label=''):
    colors = plt.cm.BuGn(np.linspace(0.5, 1, len(data)))
    fig = plt.figure()
    plt.bar(columns, data, width=0.4, color=colors)
    plt.ylabel(y_label)
    plt.title(title)
    return fig

def saveFourFigs(fig1, fig2, fig3, fig4, folder_name):
    plt.figure(fig1.number)
    plt.savefig(f'{folder_name}/number_curves.png', transparent=True, dpi=600)
    plt.figure(fig2.number)
    plt.savefig(f'{folder_name}/number_vertices.png', transparent=True, dpi=600)
    plt.figure(fig3.number)
    plt.savefig(f'{folder_name}/number_surfaces.png', transparent=True, dpi=600)
    plt.figure(fig4.number)
    plt.savefig(f'{folder_name}/number_faces.png', transparent=True, dpi=600)

def statsData2Graphs(data):
    ##curves
    columns_curves = []
    data_number = []
    data_small = []
    data_vertices = []
    for tp, d in data['curves'].items():
        columns_curves.append(tp)
        data_number.append(d['number_curves'])
        data_vertices.append(d['number_vertices'])
        # data_small.append(d['number_curves']-d['number_small_curves'])
        # data_small.append(d['number_small_curves'])
    #fig_number_curves = createNestedPieGraph(columns_curves, data_number, data_small, title='Nuber of Curves per Type')
    fig_number_curves = createPieGraph(columns_curves, data_number, title='Nuber of Curves per Type')
    fig_number_vertices = createPieGraph(columns_curves, data_vertices,  title='Nuber of Vertices per Type of Curve')

    ##surfaces
    columns_surfaces = []
    data_number = []
    data_small = []
    data_faces = [] 
    for tp, d in data['surfaces'].items():
        columns_surfaces.append(tp)
        data_number.append(d['number_surfaces'])
        data_faces.append(d['number_faces'])
        # data_small.append(d['number_surfaces']-d['number_small_surfaces'])
        # data_small.append(d['number_small_surfaces'])
    #fig_number_surfaces = createNestedPieGraph(columns_surfaces, data_number, data_small, title='Nuber of Surfaces per Type')
    fig_number_surfaces = createPieGraph(columns_surfaces, data_number, title='Nuber of Surfaces per Type')
    fig_number_faces = createPieGraph(columns_surfaces, data_faces,  title='Nuber of Faces per Type of Surface')

    return fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', type=str, help='dataset folder.')

    parser.add_argument('--stats_folder_name', type=str, default = 'stats', help='stats folder name.')
    
    parser.add_argument('--graphs_folder_name', type=str, default = 'graphs', help='graphs folder name.')
    
    args = vars(parser.parse_args())

    folder_name = args['folder']
    stats_folder_name = join(folder_name, args['stats_folder_name'])
    graphs_folder_name = join(folder_name, args['graphs_folder_name'])
    rmtree(graphs_folder_name, ignore_errors=True)
    makedirs(graphs_folder_name, exist_ok=True)

    stats_files_name = listdir(stats_folder_name)

    stats_full = {}
    for filename in tqdm(stats_files_name):
        filename = join(stats_folder_name, filename)
        name = filename[filename.rfind('/')+1:filename.rfind('.')]
        data = {}
        with open(filename, 'r') as f:
            data = json.load(f)
        model_folder_name = join(graphs_folder_name, name)
        makedirs(model_folder_name)
        fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces = statsData2Graphs(data)
        saveFourFigs(fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, model_folder_name)
        stats_full = addStats(stats_full, data)
    fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces = statsData2Graphs(stats_full)
    saveFourFigs(fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, graphs_folder_name)