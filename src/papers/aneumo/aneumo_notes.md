* Intracranial aneurysms (IAs) ~5% population, rupture = high mortality

* Risk assessment: morphology + patient-specific factors

* Hemodynamics important but unclear; CFD too slow for large-scale/clinical use

* Existing datasets limited → usually images/masks, not CFD

* Aneumo dataset:

  * 427 real aneurysm geometries
  * 10,660 synthetic 3D shapes (controlled deformation, neurosurgeon-validated)
  * 8 steady-state flow conditions (0.001–0.004 kg/s)
  * 85,280 CFD cases (velocity + pressure)
  * multimodal: 3D models, segmentation masks, point clouds, CFD outputs
  * ~4 TB public release

* CFD setup: Newtonian blood model, rigid walls, polyhedral mesh (0.15 mm), OpenFOAM solver (PISO)

* Benchmark: DeepONet vs DeepONet-SwinT (adds Swin Transformer branch)

  * DeepONet-SwinT better accuracy, generalizes to real aneurysm data
  * Inference = milliseconds per case

* Limitations: steady-state only, rigid walls; future -> transient CFD, vessel wall biomechanics
