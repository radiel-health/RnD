- Representations come from context. The paper says learned features are really about capturing the association between data and its context. For us, that context is flow metadata (Re, regime, boundary condition), while the data is geometry.

  - We split the model into a FlowTypeEncoder (for metadata) and a Geometry Encoder (for shape/coordinates). Each learns its own part, then we fuse them.

- A strength of explicit context is you can “turn the knobs.” That’s why we let log(Re) be a free input at inference—users can adjust it to explore flow behavior.

- Instead of training a new model for each dataset, we just add new metadata categories. This matches the paper’s point that expanding context variables is the right way to scale.
