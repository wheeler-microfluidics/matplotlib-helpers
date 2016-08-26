from collections import OrderedDict
import copy
import datetime as dt
import itertools

from sklearn import linear_model
import matplotlib as mpl
import numpy as np
import pandas as pd


time_total_seconds = lambda t: (60 * 60 * t.hour + 60 * t.minute + t.second
                                + 1e-6 * t.microsecond)


def time_safe(series):
    return (series.map(time_total_seconds)
            if isinstance(series.iloc[0], dt.time)
            else series)


def data_groups(df, group_key, data_key):
    return (df.set_index(group_key)[data_key].map(time_total_seconds)
            .groupby(level=0)
            if isinstance(df[data_key].iloc[0], dt.time)
            else df.groupby(group_key)[data_key])


def unique_by_column(df):
    '''
    Parameters
    ==========
    df  : pandas.DataFrame
        Data frame.

    Returns
    =======
    pandas.Series
        Mapping from each column label to ordered list of unique values in
        corresponding column in data frame.
    '''
    return pd.Series([sorted(df[column].unique()) for column in df.columns],
                     index=df.columns)


def groupif(df, key):
    if not isinstance(key, list):
        singleton_key = True
        key = [key]
    else:
        singleton_key = False

    key = pd.Series(key)

    if all([k is None for k in key]):
        if singleton_key:
            yield 0, df
        else:
            yield tuple([0] * key.size), df
    else:
        for key_i, df_i in df.groupby(key[~key.isnull()].tolist()):
            if not isinstance(key_i, (list, tuple)):
                key_i = [key_i]
            full_key_i = pd.Series(object(), index=key.index)
            full_key_i[key.isnull()] = 0
            full_key_i[~key.isnull()] = key_i
            if singleton_key:
                yield full_key_i.values[0], df_i
            else:
                yield tuple(full_key_i.values), df_i


def encode(df_data, **kwargs):
    '''
    Parameters
    ==========
    x : str
        Label of column containing ``x``-dimension.
    y  : str
        Label of column containing ``y``-dimension.
    row  : str, optional
        Label of column containing row categories.  If ``None``, all data is
        plotted in a single row of plots.
    column  : str, optional
        Label of column containing column categories. If ``None``, all data is
        plotted in a single column of plots.
    color  : str, optional
        Label of column containing color categories. If ``None``, all data is
        plotted in the same color.
    shape  : str, optional
        Label of column containing shape categories. If ``None``, all data is
        plotted using the same marker shape.
    style  : str, optional
        Label of column containing style categories. If ``None``, all data is
        plotted using the same line style.
    sharexscale  : bool or 'column'
        If ``True`` (default) all subplots share the same scale on the ``x``
        axis. If ``'column'`` all subplots *in the same column* share the same
        ``x`` axis.  If ``False``, the ``x`` axis of each subplot is scaled
        independently.
    shareyscale  : bool or 'row'
        If ``True`` (default) all subplots share the same scale on the ``y``
        axis. If ``'row'`` all subplots *in the same row* share the same ``y``
        axis.  If ``False``, the ``y`` axis of each subplot is scaled
        independently.

    Returns
    -------
    (fig, axes, keys, values)
        The ``matplotlib`` figure (``fig``), a nested dictionary (``axes``)
        indexed by row key then by column key, a ``pandas.Series`` (``keys``)
        mapping each categorical argument name to the corresponding column
        label, a ``pandas.Series`` (``values``) mapping each categorical
        argument name to a corresponding list of unique category values.
    '''
    categorical = 'row', 'column', 'color', 'shape', 'style'

    # Get column label/key associated with each category (e.g., row, column).
    keys = pd.Series([kwargs.get(k) for k in categorical], index=categorical)

    # For each category (e.g., row, column) get ordered list of unique values.
    df = df_data[keys[~keys.isnull()].unique().tolist()]
    unique_by_column_i = unique_by_column(df)
    values = pd.Series([unique_by_column_i.get(keys.get(category_i))
                        for category_i in categorical], index=categorical)

    # Find row, column, x and y range limits.
    descriptions = pd.Series()
    if keys.row is None:
        descriptions['row'] = None
    else:
        df_i = df_data
        if kwargs.get('logy'):
            df_i = df_data.loc[df_data[kwargs['y']] > 0]
        groups = data_groups(df_i, keys.row, kwargs['y'])
        descriptions['row'] = groups.describe()

    if keys.column is None:
        descriptions['column'] = None
    else:
        df_i = df_data
        if kwargs.get('logx'):
            df_i = df_data.loc[df_data[kwargs['x']] > 0]
        groups = data_groups(df_i, keys.column, kwargs['x'])
        descriptions['column'] = groups.describe()

    for axis_type_i in 'xy':
        series_i = time_safe(df_data[kwargs[axis_type_i]])
        if kwargs.get('log' + axis_type_i):
            series_i = series_i[series_i > 0]
        descriptions[axis_type_i] = series_i.describe()

    counts = values.map(lambda v: 1 if v is None else len(v))

    # extra column for legend
    grid = mpl.gridspec.GridSpec(counts.row, counts.column + 1)

    cell_size = kwargs.get('cell_size', 3)
    cell_width = kwargs.get('cell_width', cell_size)
    cell_height = kwargs.get('cell_height', cell_size)

    fig = mpl.pyplot.figure(figsize=(cell_width * (counts.column + 1),
                                     cell_height * counts.row))

    axes = OrderedDict([(row_i,
                         OrderedDict([(column_j, fig.add_subplot(grid[i, j]))
                                      for j, column_j in
                                      enumerate([0] if values.column is None
                                                else values.column)]))
                        for i, row_i in enumerate([0] if values.row is None
                                                  else values.row)])

    axis = axes.values()[0].values()[0]
    colors = OrderedDict(zip(values.color or [0],
                             itertools.imap(lambda v: v['color'],
                                            axis._get_lines.prop_cycler)))
    filled_markers = set(mpl.markers.MarkerStyle.filled_markers)
    nonfilled_markers = reversed(filter(lambda v: (v is not None) and (v !=
                                                                       'None')
                                        and (v not in filled_markers),
                                        mpl.markers.MarkerStyle.markers
                                        .keys()))
    markers = OrderedDict(zip(values['shape'] or [0],
                              itertools.cycle(itertools
                                              .chain(mpl.markers.MarkerStyle
                                                     .filled_markers,
                                                     nonfilled_markers))))

    result = {}
    if 'regression' in kwargs:
        result['regression_model'] = {}

    for (row_i, column_i), df_i in groupif(df_data, [keys.row, keys.column]):
        axis_ij = axes[row_i][column_i]
        for (color_j, shape_j), df_j in groupif(df_i, [keys.color,
                                                       keys['shape']]):
            if 'regression' in kwargs:
                model = kwargs['regression'].get('model',
                                                 linear_model
                                                 .LinearRegression())
                N = kwargs['regression'].get('N', 10)

                # Plot regression.
                X = time_safe(df_j[kwargs['x']]).values[:, np.newaxis]
                y = time_safe(df_j[kwargs['y']]).values

                if X.shape[0] > 1:
                    model.fit(X, y)

                    X_fit = np.linspace(X.ravel().min(), X.ravel().max(),
                                        N).reshape(-1, 1)
                    y_fit = model.predict(X_fit)

                    axis_ij.plot(X_fit, y_fit, color=colors[color_j],
                                 linestyle='--')
                    regress_row = (result['regression_model']
                                   .setdefault(row_i, {}))
                    regress_column = regress_row.setdefault(column_i, {})
                    regress_color = regress_column.setdefault(color_j, {})
                    regress_leaf = regress_color
                    regress_leaf[shape_j] = copy.deepcopy(model)

            # Plot markers.
            axis_ij.plot(df_j[kwargs['x']].values, df_j[kwargs['y']].values,
                         linestyle='none', marker=markers[shape_j],
                         markeredgecolor=colors[color_j]
                         if kwargs.get('stroke', True) else 'none',
                         markerfacecolor=colors[color_j]
                         if kwargs.get('fill', True) else 'none')
        axis_ij.set_xlabel(kwargs['x'])
        axis_ij.set_ylabel(kwargs['y'])

    sharexscale = kwargs.get('sharexscale', True)
    shareyscale = kwargs.get('shareyscale', True)

    for row_i, column_i, axis_i in [(row, column, v)
                                    for row, d in axes.iteritems()
                                    for column, v in d.iteritems()]:
        if kwargs.get('logx'):
            axis_i.set_xscale('log')
        if kwargs.get('logy'):
            axis_i.set_yscale('log')
        for tick in axis_i.get_xticklabels():
            tick.set_rotation(90)

        if sharexscale:
            if sharexscale == 'column':
                xmin, xmax = descriptions.column[column_i][['min', 'max']]
            else:
                xmin, xmax = descriptions.x[['min', 'max']]
            if kwargs.get('logx'):
                axis_i.set_xlim((10 ** np.floor(np.log10(xmin))),
                                (10 ** np.ceil(np.log10(xmax))))
            else:
                xspan = xmax - xmin
                axis_i.set_xlim(xmin - .05 * xspan, xmax + .05 * xspan)
        if shareyscale:
            if shareyscale == 'row':
                ymin, ymax = descriptions.row[row_i][['min', 'max']]
            else:
                ymin, ymax = descriptions.y[['min', 'max']]
            if kwargs.get('logy'):
                axis_i.set_ylim((10 ** np.floor(np.log10(ymin))),
                                (10 ** np.ceil(np.log10(ymax))))
            else:
                yspan = ymax - ymin
                axis_i.set_ylim(ymin - .05 * yspan, ymax + .05 * yspan)

    if values.column is not None:
        for i, column_i in enumerate(values.column):
            row_j = 0 if values.row is None else values.row[0]
            axis = axes[row_j][column_i]
            axis.set_title(column_i)

            if values.row is not None:
                for row_j in values.row[:-1]:
                    if sharexscale:
                        axis = axes[row_j][column_i]
                        axis.set_xlabel('')
                        axis.set_xticklabels([])

    if values.row is not None:
        for i, row_i in enumerate(values.row):
            column_j = 0 if values.column is None else values.column[0]
            axis = axes[row_i][column_j]
            axis.set_ylabel(row_i)

            if values.column is not None:
                for column_j in values.column[1:]:
                    if shareyscale:
                        axis = axes[row_i][column_j]
                        axis.set_ylabel('')
                        axis.set_yticklabels([])

    axis_legend = fig.add_subplot(grid[:, -1])

    legend_symbols = []
    legend_labels = []

    if keys['shape']:
        legend_symbols += [mpl.pyplot.Line2D([0], [0], linestyle='')]
        legend_symbols += [mpl.pyplot.Line2D([0], [0], linestyle='',
                                             marker=markers[k],
                                             color=colors[k]
                                             if keys['shape'] == keys.color
                                             else 'black')
                           for k in markers.keys()]
        legend_labels += [keys['shape']]
        legend_labels += map(lambda k: str(k).split('T')[0],
                             markers.keys())

    if keys['color'] and (keys['shape'] != keys.color):
        if keys['shape']:
            legend_symbols += [mpl.pyplot.Line2D([0], [0], linestyle='')]
            legend_labels += ['']
        legend_symbols += [mpl.pyplot.Line2D([0], [0], linestyle='')]
        legend_symbols += [mpl.pyplot.Line2D([0], [0], linestyle='',
                                             marker='s', color=colors[k])
                           for k in colors.keys()]
        legend_labels += [keys['color']]
        legend_labels += map(lambda k: str(k).split('T')[0],
                             colors.keys())

    axis_legend.legend(legend_symbols, legend_labels, loc='upper left')
    axis_legend.set_axis_off()

    fig.tight_layout()

    result.update({'fig': fig, 'axes': axes, 'keys': keys, 'values': values})
    return result


class Chart(object):
    def __init__(self, df):
        self.df = df

    def encode(self, **kwargs):
        return encode(self.df, **kwargs)
