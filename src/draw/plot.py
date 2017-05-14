#!/usr/bin/env python
import inspect
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
from parse_log import parse_log

RPN_FIELDS=    [
        ['NumIters', 'loss_cls'],
        ['NumIters', 'loss_bbox']
    ]

FAST_RCNN_FIELDS=[
            ['NumIters', 'loss_bbox'],
            ['NumIters', 'loss_cls']
        ]

FIELDS = [RPN_FIELDS, FAST_RCNN_FIELDS, RPN_FIELDS, FAST_RCNN_FIELDS]
LABELS = ["Stage 1 : RPN", "Stage 1 : Faster-RCNN", "Stage 2 : RPN", "Stage 2 : Faster-RCNN"]

def get_log_file_suffix():
    return '.log'

def get_chart_type_description_separator():
    return '  vs. '

def is_x_axis_field(field):
    x_axis_fields = ['Iters', 'Seconds']
    return field in x_axis_fields

def create_field_index():
    train_key = 'Train'
    test_key = 'Test'
    field_index = {train_key:{'NumIters':0, 'Seconds':1, train_key + ' loss':2,
                              train_key + ' learning rate':3, 'rpn_cls_loss': 4, 'rpn_loss_bbox': 5},
                   test_key:{'NumIters':0, 'Seconds':1, test_key + ' accuracy':2,
                             test_key + ' loss':3}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields

def get_supported_chart_types():
    field_index, fields = create_field_index()
    num_fields = len(fields)
    supported_chart_types = []
    for i in xrange(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in xrange(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (
                        fields[i], get_chart_type_description_separator(),
                        fields[j]))
    return supported_chart_types

def get_chart_type_description(chart_type):
    supported_chart_types = get_supported_chart_types()
    chart_type_description = supported_chart_types[chart_type]
    return chart_type_description

def get_data_file_type(chart_type):
    description = get_chart_type_description(chart_type)
    data_file_type = description.split()[0]
    return data_file_type

def get_data_file(path_to_log):
    return os.path.basename(path_to_log) + '.train';

def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(
        get_chart_type_description_separator())
    print 'description:'
    print description
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field

def get_field_indices(file):
    indices= {}
    idx=0
    with open(file, 'r') as f:
        first_line=f.readline().strip()
        for col_label in first_line.split(','):
            indices[col_label]=idx
            idx=idx+1
    return indices

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        f.readline() #skip column labels
        for line in f:

            line = line.strip()
            if line[0] != '#':
                fields = line.split(',')
                data[0].append(float(fields[field_idx0].strip()))
                data[1].append(float(fields[field_idx1].strip()))
    return data

def random_marker():
    markers = mks.MarkerStyle.markers
    num = len(markers.keys())
    idx = random.randint(0, num - 1)
    return markers.keys()[idx]

def get_data_label(path_to_log):
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())]
    return label

def get_legend_loc(chart_type):
    x_axis, y_axis = get_field_descriptions(chart_type)
    loc = 'lower right'
    if y_axis.find('accuracy') != -1:
        pass
    if y_axis.find('loss') != -1 or y_axis.find('learning rate') != -1:
        loc = 'upper right'
    return loc

def plot_chart(log_file, path_to_png):

    mean_ap=0
    phases, detected_mean_ap = parse_log(log_file)
    if detected_mean_ap != None:
        mean_ap=detected_mean_ap

    print "Processing %s with mAP=%f" % (path_to_png, mean_ap)

    plt.figure(1, figsize=(8, 32))

    end_phase=min(len(phases), 4)
    for phase_idx in range(0,end_phase):
        phase=phases[phase_idx]
        plt.subplot(411+phase_idx)
        label = LABELS[phase_idx]
        plt.title("%s%s"%( "mAP = %f    "%mean_ap if phase_idx == 0 else "",str(label[phase_idx])))
        for x_label,y_label in FIELDS[phase_idx]:
            ## TODO: more systematic color cycle for lines
            color = [random.random(), random.random(), random.random()]
            linewidth = 0.75
            ## If there too many datapoints, do not use marker.
    ##        use_marker = False
            use_marker = True
            x_data = [row[x_label] for row in phase]
            y_data = [row[y_label] for row in phase]
            if not use_marker:
                plt.plot(x_data, y_data, label = label, color = color,
                         linewidth = linewidth)
            else:
                marker = random_marker()
                plt.plot(x_data, y_data, label = label, color = color,
                         marker = marker, linewidth = linewidth)
    #legend_loc = get_legend_loc(chart_type)
    #plt.legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    #plt.xlabel(x_axis_field)
    #plt.ylabel(y_axis_field)

    # plt.annotate(fontsize='xx-small')
    print "Saving...",
    plt.savefig(path_to_png, dpi=600)
    print "done"
    plt.show()

def print_help():
    print """
    Usage: ./plot.py [log file] [output picture]
    """
    sys.exit()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_help()
    else:
        log_file = sys.argv[1]
        path_to_png = sys.argv[2]
        if not os.path.exists(log_file):
            print 'Log file does not exist: %s' % log_file
            sys.exit()
        plot_chart(log_file, path_to_png)

