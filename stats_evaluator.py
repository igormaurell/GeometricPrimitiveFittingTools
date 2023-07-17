import argparse

from tqdm import tqdm

import json

import numpy as np

from shutil import rmtree
from os import listdir, makedirs
from os.path import join

import matplotlib.pyplot as plt

def generateErrorsGraph(data_full, tp):
    assert tp in ['surfaces', 'curves']

    np.random.seed(19680801)

    x = []
    y = []
    area = []
    for data in data_full:
        if tp == 'surfaces':
            x.append(data['number_surfaces'])
            y.append(data['number_faces'])
            area.append(data['surfaces']['area'])
        elif tp == 'curves':
            x.append(data['number_curves'])
            y.append(data['number_vertices'])
            area.append(data['surfaces']['area'])

    sizes = np.ones(len(x))*200

    fig, ax = plt.subplots(figsize=(10.5, 6))
    scatter = ax.scatter(x, y, s=sizes.tolist(), c=area, cmap='jet')

    cbar = fig.colorbar(scatter, cax=axins1, orientation='vertical', ticks=[below, above])
    cbar.ax.set_xticklabels(['25', '75'])

    ax.set_xlabel('Número de Entidades Geométricas', fontsize=12)
    ax.set_ylabel('Número de Triângulos da Malha', fontsize=12)

    ax.grid(True)
    fig.tight_layout()

    plt.show()
    return fig


def addStats(dict1, dict2):
    for key, value in dict2.items():
        if type(value) is int or type(value) is float:
            if key in dict1:
                dict1[key] += value
            else:
                dict1[key] = value
        if type(value) is dict:
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

    ax.set(title=title, loc='top', bbox_to_anchor=(0.6, 0, 0.5, 1))

    ax.legend(wedges[0], labels,
            title="Types",
            loc="lower right",
            bbox_to_anchor=(0.6, 0, 0.5, 1), fontsize='large')

    return fig

def createPieGraph(labels, data, title='', num_models=1):
    fig, ax = plt.subplots(figsize=(10.5, 6))

    size = 0.8

    cmap = plt.get_cmap("tab20")
    colors_array = np.arange(len(data))*2
    colors = cmap(colors_array)

    wedges = ax.pie(data, radius=1.1, colors=colors, wedgeprops=dict(width=size, edgecolor='w'))

    
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

    s = sum(data)
    for i, p in enumerate(wedges[0]):
        percent = data[i]/s
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        if ang == 180 or ang == 0 or ang == 360:
            ang = 0.1
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(f'{round(percent*100, 2)}% (\u03BC = {data[i]//num_models})', xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set(aspect="equal", )
    #ax.set_title(title, pad=32.0, fontsize=20)
    
    ax.legend(wedges[0], labels,
            title="Types",
            loc="lower right",
            bbox_to_anchor=(1.118, -0.15, 0.5, 1), fontsize='large')
    
    return fig

def createBarGraph(columns, data, title='', y_label=''):
    colors = plt.cm.BuGn(np.linspace(0.5, 1, len(data)))
    fig = plt.figure()
    plt.bar(columns, data, width=0.4, color=colors)
    plt.ylabel(y_label)
    plt.title(title)
    return fig

def saveFigs(fig1, fig2, fig3, fig4, fig5, folder_name):
    plt.figure(fig1.number)
    plt.savefig(f'{folder_name}/number_curves.png', transparent=True, dpi=600)
    plt.close()
    plt.figure(fig2.number)
    plt.savefig(f'{folder_name}/number_vertices.png', transparent=True, dpi=600)
    plt.close()
    plt.figure(fig3.number)
    plt.savefig(f'{folder_name}/number_surfaces.png', transparent=True, dpi=600)
    plt.close()
    plt.figure(fig4.number)
    plt.savefig(f'{folder_name}/number_faces.png', transparent=True, dpi=600)
    plt.close()
    plt.figure(fig5.number)
    plt.savefig(f'{folder_name}/area_surfaces.png', transparent=True, dpi=600)
    plt.close()

def statsData2Graphs(data, num_models=1):
    ##curves
    columns_curves = []
    data_number = []
    data_vertices = []
    for tp, d in data['curves'].items():
        if type(d) == dict:
            columns_curves.append(tp)
            data_number.append(d['number_curves'])
            data_vertices.append(d['number_vertices'])
            # data_small.append(d['number_curves']-d['number_small_curves'])
            # data_small.append(d['number_small_curves'])
    #fig_number_curves = createNestedPieGraph(columns_curves, data_number, data_small, title='Nuber of Curves per Type')
    fig_number_curves = createPieGraph(columns_curves, data_number, title='Number of Curves per Type', num_models=num_models)
    fig_number_vertices = createPieGraph(columns_curves, data_vertices,  title='Number of Vertices per Type of Curve', num_models=num_models)

    ##surfaces
    columns_surfaces = []
    data_number = []
    data_faces = []
    data_area = []
    for tp, d in data['surfaces'].items():
        if type(d) == dict:
            columns_surfaces.append(tp)
            data_number.append(d['number_surfaces'])
            data_faces.append(d['number_faces'])
            data_area.append(d['area'])
        # data_small.append(d['number_surfaces']-d['number_small_surfaces'])
        # data_small.append(d['number_small_surfaces'])
    #fig_number_surfaces = createNestedPieGraph(columns_surfaces, data_number, data_small, title='Nuber of Surfaces per Type')
    fig_number_surfaces = createPieGraph(columns_surfaces, data_number, title='Number of Surfaces per Type', num_models=num_models)
    fig_number_faces = createPieGraph(columns_surfaces, data_faces,  title='Number of Faces per Type of Surface', num_models=num_models)
    fig_area = createPieGraph(columns_surfaces, data_area,  title='Area per Type of Surface', num_models=num_models)

    return fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area

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

    data_full = []
    i = 0
    for filename in tqdm(stats_files_name):
        filename = join(stats_folder_name, filename)
        name = filename[filename.rfind('/')+1:filename.rfind('.')]
        data = {}
        with open(filename, 'r') as f:
            data = json.load(f)
        model_folder_name = join(graphs_folder_name, name)
        makedirs(model_folder_name)
        fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area = statsData2Graphs(data)
        saveFigs(fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area, model_folder_name)
        data_full.append(data)
        i+= 1
    stats_full = {}
    for data in data_full:
        stats_full = addStats(stats_full, data)
    fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area = statsData2Graphs(stats_full, num_models=len(stats_files_name))
    #fig_box_plot = generateErrorsGraph(data_full, 'surfaces')
    saveFigs(fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area, graphs_folder_name)