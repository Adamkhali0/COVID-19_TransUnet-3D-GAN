# COVID-19_TransUnet-3D-GAN

# Pipeline

The proposed pipeline has three phases:

1. **Training a 3D GAN to generate synthesized healthy images:**
   - To train the generator, we use a multi-objective loss. The standard GAN objective loss (blue part) is employed to remove infected regions from COVID-19 CT images.
   - Reconstruction of healthy images from healthy-noisy image loss (green part) is utilized to generate the same structure of the lung as the input image.

2. **Build the pseudo-mask:**
   - We extract the 3D infected area using pixel-to-pixel subtraction of the synthesized healthy image from the original COVID-19 image.

3. **Developing a contrastive-aware segmentation model:**
   - To predict infected areas in 2D slices, we train a multi-objective segmentation model.
   - The model has been trained using an end-to-end pixel-wise MSE loss (blue part) and contrastive loss (green part) which is applied to the encoder only.

## Pipeline

Weights of trained models in our pipeline:

| Description                  | Link to weights               |
|------------------------------|------------------------------|
| Trained 3D Generator          | [Download](link)              |
| Trained 3D Discriminator      | [Download](link)              |
| Trained 2D-Segmentor          | [Download](link)              |

## Description of the codes

### Preprocessing

The code `Preprocess.ipynb` is used to prepare the files from raw CT scans in Nifti format to Numpy files used for training the GAN. This step consists of three main parts:

1. Load the images from Nifti format and limit the pixel value to a range of (-1000,400) based on HU (Hounsfield Unit).
2. Set the Pixel-Spacing to [1,1,1].
3. Extract the lung area from the whole image using the LungMask open-source library.
4. Reduce the size of the lung in the center of the image to 32×128×128 by cropping and resizing.
5. Save it in .npy format for future use.

The preprocessed data used to train and evaluate our model are listed below:

1. Mosmed
2. Radiopedia
3. Cronacases
4. UESTC (Note: We don't have permission to share this dataset in our GitHub based on our consent. Users need to sign a consent with the owner to access this dataset.)

### Training the GAN

To train the GAN (`GAN3D.ipynb`), we use preprocessed Healthy and Covid-19 cases. Save `LungImages.npy` for healthy (as `train_control` in the code) and Covid-19 (as `train_covid` in the code) cases separately.

```python
# In code:
train_covid = np.load('.../train_covid_cropped.npy')  # Input directory for covid cases
train_control = np.load('.../train_control_cropped.npy')  # Input directory for healthy cases
# .../generator_model: directory for saving the generator model
# .../discriminator_model: directory for saving the discriminator model
