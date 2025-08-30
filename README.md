[![Video Demo](https://img.youtube.com/vi/jNJTqcFjKK4/hqdefault.jpg)](https://youtu.be/jNJTqcFjKK4)

# Essential Vision
## Problem Statement:
Suboptimal Outcomes in AV Fistula Creation for Dialysis Patients
Patients with end-stage renal disease (ESRD) who require hemodialysis depend on the successful creation and long-term patency of arteriovenous (AV) fistulas. However, current clinical methods for selecting incision sites and predicting fistula maturation rely heavily on empirical judgment and qualitative assessments derived from ultrasound imaging. This often results in:
- High failure rates: A significant percentage of AV fistulas fail to mature properly, requiring revision surgeries or alternative access routes.

- Patient-specific variability: Vascular anatomy and flow dynamics vary significantly across individuals, making standard approaches insufficient.

- Resource-intensive workflows: Multiple imaging sessions, consultations, and revisions contribute to increased healthcare costs and patient burden.

<img width="475" height="500" alt="image" src="https://github.com/user-attachments/assets/90975508-4e44-4e00-a56f-a7365e1b0c36" />

## How Radiel Health Solves This:
Radiel Health introduces a real-time computational fluid dynamics (CFD) platform that leverages existing ultrasound imaging data to model and predict patient-specific hemodynamics before surgery. This approach addresses the limitations in current AV fistula planning by:
- Quantifying blood flow predictions: Using CFD to simulate how blood will move through potential fistula configurations, enabling evidence-based decision-making.

- Personalizing surgical planning: Tailoring incision placement recommendations to each patient’s unique vascular geometry and flow conditions.

- Streamlining the clinical workflow: Providing an intuitive platform for endocrinologists to upload ultrasound files and receive optimized surgical guidance within minutes.

- Reducing failure and revision rates: Improving the likelihood of successful fistula maturation and long-term viability through better-informed planning.

# Who it involves

## ML Team:
Rishabh
Nicholas
Sarvesh

Roles of Team Members:
Rishabh: Project Manager, Develop the GNN
Nicholas: Research, Plan out new architectures, prototype implementation
Sarvesh: Develop the GNN


## Mentor(s):
Sean Peterson (CFD Mentor)

# What constitutes success
## Team Goal:
Provide real-time Computational Fluid Dynamics (CFD) analyses to cardiac surgeons, interventional physics and device manufacturers.
## Capstone goal:
Our goal is to leverage a neural network to estimate our desired values (such as pressure or velocity of blood). This will be our job as part of the capstone project.
After predicting relevant TAWSS metrics, we've decided to make our model more generalizable by parameterizing flow conditions. This is our next step. We want to make
our model generalizable so it can be used in many situations without a significant amount of retraining: kind of a like a foundation ML model for CFD.
