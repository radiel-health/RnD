# Motivation:
Our previous model underpredicted extreme TAWSS values (> 0.05) and the results seem to get worse as the TAWSS values get larger. We want to know whether fitting a power law regression model y=ax^b would help.

# Results
TAWSS  | Linear   | RMSE = 0.0011 | R² = 0.9068
TAWSS  | Log–Log  | RMSE = 0.0010 | R² = 0.9178
TAWSS  | Non-Lin  | RMSE = 0.0010 | R² = 0.9214

Power-law estimates:
  log–log →    α = 0.554, β = 0.887  (so b≈1/β=1.127)
  non-lin →  a = 0.552, b = 1.139

# Interpretation
For log-log β = 0.887 < 1 which gives statistically evidence for the phenomenon of under-predicting high values of tawss and overestimating lower ones.

Using a power-law regression model does help for the extreme TAWSS values (> 0.05) . It also does improve overall R^2 from 0.9068 to 0.9214 (higher is better)