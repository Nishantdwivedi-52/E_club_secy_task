import pandas as pd


def create_clean_csv(attr_path, output_csv):

    df = pd.read_csv(attr_path)


    df.replace(-1, 0, inplace=True)


    if 'No_Beard' not in df.columns or 'Male' not in df.columns:
        raise KeyError("Required columns 'No_Beard' or 'Male' are missing in the dataset.")

    df['beard'] = (df['No_Beard'] == 0).astype(int)


    df = df[['image_id', 'Male', 'beard']].copy()
    df.rename(columns={'image_id': 'filename', 'Male': 'gender'}, inplace=True)


    df.to_csv(output_csv, index=False)
    print(f"[+] Saved {len(df)} records to {output_csv}")


if __name__ == "__main__":
    create_clean_csv(
        attr_path="Celeb_dataset/list_attr_celeba.csv",
        output_csv="Celeb_dataset/celeba_gender_beard.csv"
    )
import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(input_csv, train_csv, test_csv, test_size=0.2, random_state=42):

    df = pd.read_csv(input_csv)


    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['gender']
    )


    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"[+] Train set: {len(train_df)} samples → {train_csv}")
    print(f"[+] Test set:  {len(test_df)} samples → {test_csv}")

if __name__ == "__main__":
    split_csv(
        input_csv="Celeb_dataset/celeba_gender_beard.csv",
        train_csv="Celeb_dataset/train_gender_beard.csv",
        test_csv="Celeb_dataset/test_gender_beard.csv"
    )


from tensorflow.keras.preprocessing.image import ImageDataGenerator


df = pd.read_csv('Celeb_dataset/train_gender_beard.csv')
df['gender'] = df['gender'].astype(str)  # Keras expects string labels

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory='Celeb_dataset/img_align_celeba/img_align_celeba',  # path to images
    x_col='filename',
    y_col='gender',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory='Celeb_dataset/img_align_celeba/img_align_celeba',
    x_col='filename',
    y_col='gender',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

model.save('gender_classifier.h5')
print("Model saved as gender_classifier.h5")
