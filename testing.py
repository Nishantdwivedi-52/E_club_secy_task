from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = load_model('gender_classifier.h5')  # or 'gender_classifier.keras'

df_test = pd.read_csv('Celeb_dataset/test_gender_beard.csv')
df_test['gender'] = df_test['gender'].astype(str)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory='Celeb_dataset/img_align_celeba/img_align_celeba',
    x_col='filename',
    y_col='gender',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_gen)
print(f"Test accuracy: {test_acc:.4f}")

#  predict on new samples
# predictions = model.predict(test_gen)
