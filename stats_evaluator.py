import argparse

from tqdm import tqdm

import json

import numpy as np

from shutil import rmtree
from os import listdir, makedirs
from os.path import join

import matplotlib.pyplot as plt

from asGeometryOCCWrapper.curves import CurveFactory
from asGeometryOCCWrapper.surfaces import SurfaceFactory

EN_DICT = {
    'Ellipse': 'Ellipse',
    'BSpline': 'B-Spline',
    'Circle': 'Circle',
    'Line': 'Line',
    'Types': 'Types',
    'Amount': 'Amount',
    'Sphere': 'Sphere',
    'Cylinder': 'Cylinder',
    'Cone': 'Cone',
    'Torus': 'Torus',
    'BSpline': 'B-Spline',
    'Plane': 'Plane',
    'Area': 'Area',
}

PT_DICT = {
    'Ellipse': 'Elipse',
    'BSpline': 'B-Spline',
    'Circle': 'Círculo',
    'Line': 'Reta',
    'Types': 'Tipos',
    'Amount': 'Quantidade',
    'Sphere': 'Esfera',
    'Cylinder': 'Cilindro',
    'Cone': 'Cone',
    'Torus': 'Toro',
    'BSpline': 'B-Spline',
    'Plane': 'Plano',
    'Area': 'Área',
}

LANG_DICT = {
    'en': EN_DICT,
    'pt': PT_DICT,
}

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

    wedges = ax.pie(out_data, radius=1.1, colors=outer_colors, autopct=lambda pct: func(pct, in_data), fontsize='x-large', pctdistance=0.85,
        wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(in_data, radius=1.1-size, colors=inner_colors, autopct=lambda pct: func(pct, in_data), fontsize='x-large', pctdistance=1.15-size,
        wedgeprops=dict(width=size, edgecolor='w'))

    #ax.set(title=title, loc='top', bbox_to_anchor=(0.6, 0, 0.5, 1))

    labels = [LANG_DICT[LANG][l] for l in labels]

    ax.legend(wedges[0], labels,
            title=LANG_DICT[LANG]["Types"],
            loc="lower right",
            bbox_to_anchor=(0.6, 0, 0.5, 1), fontsize='xx-large')

    return fig

def createPieGraph(labels, data, title='', num_models=1, geometry_type='surface', **kwargs):
    fig, ax = plt.subplots(figsize=(20, 8))

    size = 0.8

    #cmap = plt.get_cmap("tab20")
    #colors_array = np.arange(len(data))*2
    #colors = cmap(colors_array)

    if geometry_type == 'curve':
        colors = np.asarray([list(CurveFactory.FEATURES_CURVE_CLASSES[l].getColor()) + [255] for l in labels])
    elif geometry_type == 'surface':
        colors = np.asarray([list(SurfaceFactory.FEATURES_SURFACE_CLASSES[l].getColor()) + [255] for l in labels])
    else:
        assert False

    labels = [LANG_DICT[LANG][l] for l in labels]

    colors = colors/255.

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
                    horizontalalignment=horizontalalignment, fontsize='xx-large', **kw)

    ax.set(aspect="equal", )
    #ax.set_title(title, pad=32.0, fontsize=20)
    
    plt.legend(wedges[0], labels,
            title=LANG_DICT[LANG]["Types"],
            loc="lower right",
            fontsize='x-large')
    plt.rcParams['legend.title_fontsize'] = 'xx-large'
    
    return fig

def createBarGraph(labels, data, title='', num_models=1, geometry_type='surface', data_label='Amount'):
    fig, ax = plt.subplots(figsize=(14, 8))

    if geometry_type == 'curve':
        colors = np.asarray([list(CurveFactory.FEATURES_CURVE_CLASSES[l].getColor()) + [255] for l in labels])
    elif geometry_type == 'surface':
        colors = np.asarray([list(SurfaceFactory.FEATURES_SURFACE_CLASSES[l].getColor()) + [255] for l in labels])
    else:
        assert False

    labels = [LANG_DICT[LANG][l] for l in labels]

    colors = colors/255.

    width = 0.2
    offset = 0.1

    data_sum = sum(data)
    percents = [(d/data_sum) for d in data]

    sorted_indices = sorted(range(len(labels)), key=lambda x: data[x])
    
    x = []
    for i in sorted_indices:
        last_pos = x[-1] if len(x) > 0 else 0
        pos = last_pos + width + offset
        rects = ax.bar(pos, data[i], width, label=labels[i], color=colors[i])
        ax.bar_label(rects, fmt=f'{round(percents[i]*100, 2)}% (\u03BC = {(data[i]//num_models):.1E})', padding=3) 
        #ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')
        x.append(pos)
    
    #ax.legend(loc='upper left', ncols=3)
    ax.set_ylabel(data_label, fontsize=14)
    ax.set_xlabel(LANG_DICT[LANG]["Types"], fontsize=14)
    #ax.set_title(title, pad=32.0, fontsize=20)
    ax.set_xticks(x, [labels[i] for i in sorted_indices])

    # for attribute, measurement in penguin_means.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, measurement, width, label=attribute)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1
    
    return fig

def createBarHGraph(labels, data, title='', num_models=1, geometry_type='surface', data_label='Amount'):
    fig, ax = plt.subplots(figsize=(24, 12))

    if geometry_type == 'curve':
        colors = np.asarray([list(CurveFactory.FEATURES_CURVE_CLASSES[l].getColor()) + [255] for l in labels])
    elif geometry_type == 'surface':
        colors = np.asarray([list(SurfaceFactory.FEATURES_SURFACE_CLASSES[l].getColor()) + [255] for l in labels])
    else:
        assert False

    labels = [LANG_DICT[LANG][l] for l in labels]

    colors = colors/255.

    height = 0.2
    offset = 0.1

    data_sum = sum(data)
    percents = [(d/data_sum) for d in data]

    sorted_indices = sorted(range(len(labels)), key=lambda x: data[x])
    
    y = []
    for i in sorted_indices:
        last_pos = y[-1] if len(y) > 0 else 0
        pos = last_pos + height + offset
        rects = ax.barh(pos, data[i], height, label=labels[i], color=colors[i])
        ax.bar_label(rects, fmt=f'  {round(percents[i]*100, 2)}%', padding=3, fontsize=38) 
        y.append(pos)
    
    #ax.legend(loc='upper left', ncols=3)
    ax.set_xlabel(data_label, fontsize=42, fontweight='bold')
    ax.set_ylabel(LANG_DICT[LANG]["Types"], fontsize=42, fontweight='bold')
    #ax.set_title(title, pad=32.0, fontsize=20)
    ax.set_yticks(y, [labels[i] for i in sorted_indices], fontsize=40)
    ax.tick_params(axis='x', labelsize=40)
    t = ax.xaxis.get_offset_text()
    t.set_size(40)
    #plt.xticks(fontsize=30)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.2*max(data))

    # for attribute, measurement in penguin_means.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, measurement, width, label=attribute)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1
    
    return fig

def saveFigs(fig1, fig2, fig3, fig4, fig5, folder_name):
    plt.figure(fig1.number)
    plt.savefig(f'{folder_name}/number_curves.pdf', format='pdf', transparent=True, dpi=1200)
    plt.close()
    plt.figure(fig2.number)
    plt.savefig(f'{folder_name}/number_vertices.pdf', format='pdf', transparent=True, dpi=1200)
    plt.close()
    plt.figure(fig3.number)
    plt.savefig(f'{folder_name}/number_surfaces.pdf', format='pdf', transparent=True, dpi=1200)
    plt.close()
    plt.figure(fig4.number)
    plt.savefig(f'{folder_name}/number_faces.pdf', format='pdf', transparent=True, dpi=1200)
    plt.close()
    plt.figure(fig5.number)
    plt.savefig(f'{folder_name}/area_surfaces.pdf', format='pdf', transparent=True, dpi=1200)
    plt.close()

def statsData2Graphs(data, num_models=1, graph='barh'):
    if graph=='bar':
        func = createBarGraph
    elif graph=='barh':
        func = createBarHGraph
    elif graph=='pie':
        func = createPieGraph
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
    fig_number_curves = func(columns_curves, data_number, title='Number of Curves per Type', 
                             num_models=num_models, geometry_type='curve', data_label=LANG_DICT[LANG]['Amount'])
    fig_number_vertices = func(columns_curves, data_vertices,  title='Number of Vertices per Type of Curve', 
                               num_models=num_models, geometry_type='curve', data_label=LANG_DICT[LANG]['Amount'])

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
    fig_number_surfaces = func(columns_surfaces, data_number, title='Number of Surfaces per Type',
                                         num_models=num_models, data_label=LANG_DICT[LANG]['Amount'])
    fig_number_faces = func(columns_surfaces, data_faces,  title='Number of Faces per Type of Surface',
                                      num_models=num_models, data_label=LANG_DICT[LANG]['Amount'])
    fig_area = func(columns_surfaces, data_area,  title='Area per Type of Surface',
                              num_models=num_models, data_label=LANG_DICT[LANG]['Area'])

    return fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('folder', type=str, help='dataset folder.')

    parser.add_argument('--stats_folder_name', type=str, default = 'stats', help='stats folder name.')
    parser.add_argument('--graphs_folder_name', type=str, default = 'graphs', help='graphs folder name.')
    parser.add_argument('--lang', type=str, default='en', choices=['pt', 'en'], help='language.')
    
    args = vars(parser.parse_args())

    folder_name = args['folder']
    stats_folder_name = join(folder_name, args['stats_folder_name'])
    graphs_folder_name = join(folder_name, args['graphs_folder_name'])
    rmtree(graphs_folder_name, ignore_errors=True)
    makedirs(graphs_folder_name, exist_ok=True)
    LANG = args['lang']

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
        fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area = statsData2Graphs(data)
        saveFigs(fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area, model_folder_name)
        stats_full = addStats(stats_full, data)
    fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area = statsData2Graphs(stats_full, num_models=len(stats_files_name))
    saveFigs(fig_number_curves, fig_number_vertices, fig_number_surfaces, fig_number_faces, fig_area, graphs_folder_name)