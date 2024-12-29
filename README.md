# **Car Viewpoint Classification Project**

## **Project Overview**
This project aims to classify car images into six predefined viewpoints:
- Front
- Front-Left
- Front-Right
- Rear
- Rear-Left
- Rear-Right  
Additionally, a "None" class handles ambiguous or irrelevant images, such as close-ups of isolated car parts or unrelated scenes.

The project involves creating a robust pipeline for data preprocessing, model training, and optimization for edge deployment.

---

## **Project Goals**
1. Accurately classify car viewpoints with high precision and recall across all classes.
2. Effectively handle edge cases, such as:
   - Overlapping or ambiguous viewpoints.
   - Partial views of the car.
   - Irrelevant or distorted images for the "None" class.
3. Optimize the model for deployment on resource-constrained devices.

---

## **Tasks and Milestones**

1. **Project Setup**:
   - [x] Create the GitHub repository.
   - [x] Define the project structure and dependencies.

2. **Data Preparation**:
   - [ ] Analyze and clean the dataset.
   - [ ] Develop a data loader and augmentation pipeline.

3. **Model Development**:
   - [ ] Design the initial model and implement training and validation loops.
   - [ ] Experiment with different architectures and hyperparameters.

4. **Iterative Refinement**:
   - [ ] Address class imbalances and improve performance on edge cases.
   - [ ] Fine-tune augmentations and model configurations.

5. **Model Optimization**:
   - [ ] Quantize and optimize the model for edge deployment.
   - [ ] Test for latency and resource usage on target devices.

---

## **Folder Structure**
```
/data - Raw and processed datasets
/notebooks - Data exploration and analysis notebooks
/src - Core code (dataloader, model, training, etc.)
/configs - Configuration files for hyperparameters
/outputs - Logs, model checkpoints, and predictions
/scripts - Utility scripts for setup, preprocessing, etc.
```

---

## **Key Challenges**
- Handling ambiguous and overlapping viewpoints.
- Ensuring robust performance for the "None" class with limited data.
- Optimizing the model for real-time inference on edge devices.

---

## **Requirements**
- Python (>=3.8)
- Key libraries:
  - PyTorch / TensorFlow
  - Albumentations for augmentations
  - Matplotlib/Seaborn for data visualization
- Additional dependencies are listed in `requirements.txt`.

---

## **How to Run**
1. Clone the repository and set up the environment.
2. Prepare the dataset by parsing annotations and creating splits.
3. Train the model:
   - Modify hyperparameters in `configs/config.yaml`.
   - Run the training script in `/src`.
4. Evaluate the model using the validation and test datasets.

---

## **Future Scope**
- Extend the model for additional car viewpoints or object detection.
- Incorporate real-time feedback mechanisms for further deployment enhancements.





