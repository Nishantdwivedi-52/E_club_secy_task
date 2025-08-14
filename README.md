# E_club_secy_task
STEPS INVOLVED 
1. Preprocessing the Dataset
I started by reading the list_attr_celeba.csv file, which contains attributes for each image in the CelebA datasetwhich i took from Kaggle 

All values of -1 were replaced with 0 to standardize the binary labels.

I created a new binary column named beard, derived from the No_Beard column (beard = 1 if the person has a beard).

I filtered the dataset to keep only the columns: image_id, Male, and beard.

Renamed the columns to more intuitive names: image_id → filename and Male → gender.

The cleaned data was saved into a new CSV file: celeba_gender_beard.csv.

2. Splitting the Dataset
I split the cleaned CSV into training and testing sets using an 80/20 ratio.

Stratified splitting was used to ensure gender balance in both train and test datasets.

The output files were saved as train_gender_beard.csv and test_gender_beard.csv.

3. Data Preparation for Model Training
I used ImageDataGenerator to load and preprocess the image data.

Applied normalization (rescaling pixel values to between 0 and 1).

Split the training data further into training and validation sets  .

Created training and validation data generators that read images from disk and pair them with gender labels.

4. Model Building and Training
I designed a simple CNN using TensorFlow/Keras:
 

Trained the model for 5 epochs using the training and validation data generators.

5. Saving the Model
After training, I saved the trained model to a file named gender_classifier.h5 to use it in testing.py file

IN TESTING 

6. Model Evaluation
Loaded the saved model using load_model().

Prepared the test dataset (test_gender_beard.csv) with ImageDataGenerator  .

Evaluated the model's performance on the test set using model.evaluate().

Printed the final test accuracy, which measures how well the model generalizes to unseen data.



7.Prepared Data for CycleGAN

Filtered male images from the dataset.

Created two folders: one for males without beards, one for males with beards.

Copied images to corresponding folders.

8. Trained a CycleGAN Model

Ran the training script in the terminal using the CycleGAN repo.

Used 1000 images and trained for 2 epochs.
9 . Run this in terminal to get beared2nobeared file 
python train.py --dataroot ../cycle_data --name beard2nobeard --model cycle_gan \
--load_size 128 --crop_size 128 --batch_size 4 --gpu_ids 0 \
--n_epochs 1 --n_epochs_decay 1 --max_dataset_size 1000 --display_id 0



NOTES 

i used cycleGAN repo for beared removal and addition 

i treained the cycleGAN only for 1000 images and 2 epoch because this only take me approx. 2 hrs traning on big dataset was not feasible for me 

although i trained male and female detection on whole celeb dataset that took 2 hr approx. too 
     
by traning only on 1000 images the end results were not satisfactory but it can be improved just we have to increae the traning dataset  


also i can't submit my pictures file like beared2nobeared or male female because those are very big files 

Due to time constraints i can't do the final step that use an input image and test weather it is a male or female and then apply the bearded addition and removal model 
