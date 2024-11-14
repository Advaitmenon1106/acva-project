# ACVA-project

## Brain tumor clssification and segmentation

### Important files:-

- App2.py: The final streamlit output
- best_model.pt: The model that is fine-tuned on the brain tumor dataset
- Autoencoder file: Autoencoder generalised on the ADE20K, this was finetuned on the brain dataset, but led to no results
- Code.ipynb: Raw code file used only for some working- training code for YOLO and training code for the autoencoder
- datasets: YOLO-maintained dataset for brain tumor detection. Includes images and labels for training and validation data
- vgg16_functional_model.h5: VGG model for tumor prediction
- predictions3.py: Pipeline for the VGG model
