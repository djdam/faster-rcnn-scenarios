#!/usr/bin/env python

"""
Parse training log

Evolved from py-faster-rcnn's parse_log.
"""

import os
import re
import extract_seconds
import argparse
import csv
from collections import OrderedDict

def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, test_dict_list)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows
    """
    regex_float = '([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)'
    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_learning_rate = re.compile('lr = %s'%regex_float)
    regex_end_of_phase = re.compile('Wrote snapshot to')
    regex_ignore_rows = re.compile('speed: [0-9\.]*s / iter')
    regex_mean_ap = re.compile('Mean AP = %s'%regex_float)
    # Pick out lines of interest
    iteration = -1
    learning_rate = float('NaN')
    phases=[]
    train_dict_list=[]


    train_row = None
    test_row = None

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    mean_ap =  None

    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)
        last_time = start_time

        for line in f:

            if regex_ignore_rows.search(line):
                continue

            if regex_end_of_phase.search(line):
                # start a new learning phase
                fix_initial_nan_learning_rate(train_dict_list)
                phases.append(train_dict_list)
                train_dict_list=[]
                iteration=-1
                continue

            mean_ap_match=regex_mean_ap.search(line)
            if mean_ap_match:
                mean_ap=float(mean_ap_match.group(1))

            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
                continue
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            try:
                time = extract_seconds.extract_datetime_from_line(line,
                                                                  logfile_year)
            except ValueError:
                # Skip lines with bad formatting, for example when resuming solver
                continue

            # if it's another year
            if time.month < last_time.month:
                logfile_year += 1
                time = extract_seconds.extract_datetime_from_line(line, logfile_year)
            last_time = time

            seconds = (time - start_time).total_seconds()

            learning_rate_match = regex_learning_rate.search(line)
            if learning_rate_match:
                learning_rate = float(learning_rate_match.group(1))
                continue

            train_dict_list, train_row, train_match = parse_line_for_net_output(
                regex_train_output, train_row, train_dict_list,
                line, iteration, seconds, learning_rate
            )

    return phases, mean_ap


def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, seconds, learning_rate):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """

    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('NumIters', iteration),
                ('Seconds', seconds),
                ('LearningRate', learning_rate)
            ])

        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)

    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row, output_match


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """

    if len(dict_list) > 1:
        dict_list[0]['LearningRate'] = dict_list[1]['LearningRate']


def save_csv_files(logfile_path, output_dir, train_dict_list, test_dict_list,
                   delimiter=',', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    train_filename = os.path.join(output_dir, log_basename + '.train')
    write_csv(train_filename, train_dict_list, delimiter, verbose)

    test_filename = os.path.join(output_dir, log_basename + '.test')
    write_csv(test_filename, test_dict_list, delimiter, verbose)


def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_dict_list, test_dict_list = parse_log(args.logfile_path)
    save_csv_files(args.logfile_path, args.output_dir, train_dict_list,
                   test_dict_list, delimiter=args.delimiter, verbose=args.verbose)


if __name__ == '__main__':
    main()
