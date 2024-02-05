"""

As sensitivity analysis tool, tornado diagrams provide an intuitive and easy-to-interpret
graphical visualization of a series of one-factor-at-a-time (OFAT) sensitivities.
They therefore facilitate communication between analysts and decision makers.

In terms of computational cost, a tornado diagram requires 2n + 1 model runs.

In this example we use a trained model from primary_model (which can be replaced by another
variable) package and CaliforniaHousingDataset (which can also be replaced easily by another
dataset).

Using the above two, we set a base case point (random instance) from dataset.
Secondly, we use the dataset to fnd the boundary conditions to evaluate each variable.
Here, boundary conditions are minimum and maximum of each variable based on the dataset
values. However, this could be changed by any values (hard-coded).

Lastly, we run model (model_inference) on the base_case and boundary conditions to get
model response. Later, the model response on base_case is subtracted from model
response on boundary_condition. This difference/impact is plotted in 2 tornado diagrams.

"""