Usage
-----

The examples below plot vehicle fuel economy (in miles per gallon) versus
horsepower for a dataset from the [altair][1] project.

Set marker color by the `Year` column and set the shape of the each marker
according to the `Origin` column:

```python
    from altair import load_dataset
    import matplotlib as mpl
    import matplotlib.style
    import matplotlib_helpers as mplh
    import matplotlib_helpers.chart

    # load data as a pandas DataFrame
    cars = load_dataset('cars')

    with mpl.style.context(['ggplot']):
        mplh.chart.encode(cars,
                          x='Horsepower',
                          y='Miles_per_Gallon',
                          shape='Year',
                          color='Origin', 
                          cell_size=5, fill=False)
```

![Fuel economy vs horsepower][plot]

Split plot into multiple subplots, with the subplot in each column
corresponding to a distinct value in the `Origin` column.

The same type of handling can be applied using the `row` keyword.

```python
    with mpl.style.context(['ggplot']):
        mplh.chart.encode(cars,
                          x='Horsepower',
                          y='Miles_per_Gallon',
                          color='Year',
                          shape='Year',
                          column='Origin', 
                          cell_size=5, fill=False)
```

![Fuel economy vs horsepower (columns by "Origin")][column-plot]

By default, all plots share the same `x` axis scale and `y` axis scale.  This
behaviour can be changed by setting the `sharexscale` keyword argument or the
`shareyscale` keyword argument.

For example, note that the subplots below all have different `x` axis and `y`
axis scales.

```python
    with mpl.style.context(['ggplot']):
        mplh.chart.encode(cars,
                          x='Horsepower',
                          y='Miles_per_Gallon',
                          color='Year',
                          shape='Year',
                          column='Origin', 
                          sharexscale=False,
                          shareyscale=False,
                          cell_size=5, fill=False)
```

![Fuel economy vs horsepower (axis scales not shared)][column-plot-shareyaxis_false]


[1]: https://github.com/ellisonbg/altair
[plot]: docs/fuel_economy-vs-horsepower.png
[column-plot]: docs/fuel_economy-vs-horsepower-origin_columns.png
[column-plot-shareyaxis_false]: docs/fuel_economy-vs-horsepower-origin_columns-shareyaxis_false.png
