mpnn_pipeline_wss_and_osi_v3_deep.ipynb is the oldest model. It combines many GNN layers so that the resulting network is "deep". This is the earliest
mpnn_pipeline_tawss_unet.ipynb is the newer model that aggregates data using a node graph to line graph method. This is newer.
mpnn_pipeline_tawss_unet_residual_calibration.ipynb is a model that uses our team's research in quantifying uncertainty and recallibrating the data to improve the previous model.
fold_best_states are the model weights that we trained. They are assumed to be uploaded in the notebook env.
data.zip has the raw CFD simulation results that we used for training.
