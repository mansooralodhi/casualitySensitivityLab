"""
In certain circumstances, an analyst might be interested in the response of the model when the
model inputs vary between their extreme ranges.
When we inspect the behavior of the _artifacts as we vary one input within a predetermined
range, we obtain a one-way sensitivity function.
Thus, for a model input x_i , we have as many one-way sensitivity functions as the number of values that we
can assign to the other model inputs (infinitely many, in general).

The number of model evaluations necessary to obtain a spider plot is
C_OneWay = n * N.

"""