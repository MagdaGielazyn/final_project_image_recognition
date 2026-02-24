import hashlib
import kagglehub 
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf   

from PIL import Image


#sprawdzenie balansu rozlozenia klas w zbiorze:
def check_balance(df, target):
    df_percentage = df[target].value_counts(normalize=True) * 100
    num = df_percentage.max()
    if num == 25:
        descr = "Dane są idealnie zrównoważone."
    elif num <= 35:
        descr = "Dane są dobrze zrównoważone."
    elif num <= 45:
        descr = "Lekka nierównowaga – akceptowalna."
    elif num <= 55:
        descr = "Umiarkowana nierównowaga – warto monitorować."
    else:
        descr = "Silna nierównowaga – zalecane balansowanie danych."
    return descr

#wczytanie danych:
def data_reader(path: str, img_size=(150, 150), batch_size=None):
    dataset_path = kagglehub.dataset_download(path)
    train_dir = os.path.join(dataset_path, "Training")
    test_dir = os.path.join(dataset_path, "Testing")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False, ### na razie tak zamiast False zeby sciezki podpiac
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    train_paths = tf.data.Dataset.from_tensor_slices(train_ds.file_paths)
    test_paths = tf.data.Dataset.from_tensor_slices(test_ds.file_paths)


    enriched_train_ds = tf.data.Dataset.zip((train_ds, train_paths))
    enriched_test_ds = tf.data.Dataset.zip((test_ds, test_paths))
    full_ds=enriched_train_ds.concatenate(enriched_test_ds)

    categories = set(train_ds.class_names+test_ds.class_names)
    #train_ds = train_ds.batch(batch_size)
    #test_ds = test_ds.batch(batch_size)

    return (full_ds, categories)

#zmiana datasetu na df:
def dataset_to_df(ds):
    return pd.DataFrame([
        (p.numpy().decode(), int(np.argmax(y.numpy())))
        for (x, y), ps in ds
        for p, y in zip(ps, y)
    ], columns=["path", "label"])

#znalezienie duplikatow:
def find_image_duplicates(**kwargs):
    datasets = list(kwargs.values())
    def image_hash(image):
      image_bytes = tf.io.serialize_tensor(image)
      return hashlib.md5(image_bytes.numpy()).hexdigest()
    if not datasets:
        return {
            "Liczba wszystkich obrazów:": 0,
            "Liczba duplikatów:": 0,
            "Unikalne obrazy:": 0,
            "Duplikaty szczegóły:": []
        }

    full_ds = datasets[0]
    for ds in datasets[1:]:
        full_ds = full_ds.concatenate(ds)

    hash_to_path = {}
    duplicates = []
    total_images = 0

    for (img, _), path in full_ds:
      total_images += 1
      h = image_hash(img)
      #print(h)
      path_str = path.numpy().decode()

      if h in hash_to_path:

          duplicates.append((path_str, hash_to_path[h]))
      else:

          hash_to_path[h] = path_str

    return {
        "Number of images": total_images,
        "Number of duplicates": len(duplicates),
        "Unique images": len(hash_to_path),
        "Duplicates details": duplicates
    }

#pokazanie rozdystrybuuowania klas w zbiorze:
def plot_class_distribution(df, column="label_name", title="Rozkład klas"):
    labels = df[column].value_counts().index.tolist()
    colors = sns.color_palette("Blues", n_colors=len(labels))
    
    df[column].value_counts(normalize=True).plot(
        kind="pie",
        autopct="%1.1f%%",
        labels=labels,
        colors=colors,
        title=title,
        ylabel=""
    )
    
    plt.show()

#usuniecie zduplikowanych obrazow:
def remove_paths_from_dataset(dataset, paths_to_remove):
    paths_to_remove = tf.constant(paths_to_remove)
    def keep_fn(data, path):
        is_equal = tf.reduce_any(tf.equal(paths_to_remove, path))
        return tf.logical_not(is_equal)

    return dataset.filter(keep_fn)


#pokazanie obrazow ze zbiorow:
def show_images(paths, rows=None, cols=None, figsize=(16, 8)):
    n = len(paths)
    if rows is None and cols is None:
        cols = min(5, n)
        rows = math.ceil(n / cols)
    elif rows is None:
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)

    plt.figure(figsize=figsize)

    for i, path in enumerate(paths, 1):
        img = Image.open(path)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#podzial datasetu na traningowy i testowy:
def split_train_test_ds(dataset, dataset_size, split_ratio=0.2, batch_size=32):
    dataset = dataset.shuffle(buffer_size=dataset_size)
    split_index = int(dataset_size * split_ratio)
    train_dataset = dataset.skip(split_index)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = dataset.take(split_index)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset