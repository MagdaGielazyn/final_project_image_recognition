# Import bibliotek:
import hashlib
import kagglehub
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import tensorflow as tf
from collections import Counter

from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from tensorflow.keras import layers
from scipy.ndimage import gaussian_filter

# Lista funkcji - posortowana alfabetycznie ASC:

# Zamiana batched datasetu na pandas DataFrame
def batched_dataset_to_df(ds):
    return pd.DataFrame([
        (p.numpy().decode(), int(np.argmax(y.numpy())))
        for (x, y), ps in ds
        for p, y in zip(ps, y)
    ], columns=["path", "label"])


# Sprawdzenie balansu rozkładu klas w zbiorze danych
def check_balance(df, target):
    df_percentage = df[target].value_counts(normalize=True) * 100
    num = df_percentage.max()
    if num <= 26:
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


# Analiza błędów
def collect_errors(model, dataset, normalize=False):
    errors = []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)

        for img, true_oh, pred_probs in zip(x_batch.numpy(), y_batch.numpy(), preds):
            true_label = np.argmax(true_oh)
            pred_label = np.argmax(pred_probs)

            if true_label != pred_label:
                errors.append({
                    "image": img / 255.0 if normalize else img,
                    "true": true_label,
                    "predicted": pred_label,
                    "confidence": float(pred_probs[pred_label])
                })

    return errors


# Wczytanie danych z Kaggle i utworzenie datasetów treningowego i testowego
def data_reader(path: str, img_size=(150, 150), batch_size=None):
    dataset_path = kagglehub.dataset_download(path)
    train_dir = os.path.join(dataset_path, "Training")
    test_dir = os.path.join(dataset_path, "Testing")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical", # Etykiety jako one-hot encoding
        shuffle=False  # Ścieżki pozostają w oryginalnej kolejności
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    train_paths = tf.data.Dataset.from_tensor_slices(train_ds.file_paths)
    test_paths = tf.data.Dataset.from_tensor_slices(test_ds.file_paths)

    enriched_train_ds = tf.data.Dataset.zip((train_ds, train_paths))
    enriched_test_ds = tf.data.Dataset.zip((test_ds, test_paths))
    full_ds = enriched_train_ds.concatenate(enriched_test_ds)

    categories = set(train_ds.class_names + test_ds.class_names)

    return (full_ds, categories, train_ds, test_ds)


# Zamiana datasetu na DataFrame (bez batching)
def dataset_to_df(ds):
    return pd.DataFrame([
        (path.numpy().decode(), int(np.argmax(label.numpy())))
        for (image, label), path in ds
    ], columns=["path", "label"])


# Definicja modelu EfficientNetB0 z transfer learningiem
def effNet_model(num_classes=4, input_shape=(150, 150, 3)):
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",       # wczytanie wag wytrenowanych na 1.2M zdjęć, 1000 klas
        include_top=False,        # odcięcie oryginalnego klasyfikatora ImageNet
        input_shape=input_shape
    )
    base_model.trainable = False  # zamrożenie wszystkich wag bazowych

       # Klasyfikator - podejmowanie decyzji
    x = base_model.output # mapa cech z EfficientNetB0 (4×4×1280 dla 150×150)
    x = layers.GlobalAveragePooling2D()(x) # agregacja do wektora 1280 wartości
    x = layers.Dropout(0.3)(x) # regularyzacja, zapobiega overfittingowi
    outputs = layers.Dense(num_classes, activation="softmax")(x) # 4 prawdopodobieństwa dla 4 klas guza

    return tf.keras.Model(inputs=base_model.input, outputs=outputs)


# Wykrywanie duplikatów obrazów w datasetach
def find_image_duplicates(**kwargs):
    datasets = list(kwargs.values())

    def image_hash(image):
        image_bytes = tf.io.serialize_tensor(image)
        return hashlib.md5(image_bytes.numpy()).hexdigest()

    if not datasets:
        return {
            "Number of images": 0,
            "Number of duplicates": 0,
            "Unique images": 0,
            "Duplicates details": []
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


# Generowanie tabeli metryk w markdown
def generate_markdown_table(cnn_results, effNet_results):
    f1_cnn = f1_score(cnn_results["y_true"], cnn_results["y_pred"], average="macro")
    f1_effNet = f1_score(effNet_results["y_true"], effNet_results["y_pred"], average="macro")

    df = pd.DataFrame({
        "Metryka": [
            "Test Accuracy",
            "Test Loss",
            "Macro F1-Score",
            "Liczba parametrów ogółem",
            "Liczba parametrów trenowalnych",
            "Epoki do zatrzymania"
        ],
        "CNN": [
            f"{cnn_results['accuracy']:.2%}",
            f"{cnn_results['loss']:.4f}",
            f"{f1_cnn:.2%}",
            f"{cnn_results['params']:,}",
            f"{cnn_results['params_trainable']:,}",
            len(cnn_results['history'].history['accuracy'])
        ],
        "EfficientNetB0": [
            f"{effNet_results['accuracy']:.2%}",
            f"{effNet_results['loss']:.4f}",
            f"{f1_effNet:.2%}",
            f"{effNet_results['params']:,}",
            f"{effNet_results['params_trainable']:,}",
            len(effNet_results['history'].history['accuracy'])
        ]
    })

    return df.to_markdown(index=False)


# Generowania mapy aktywacji Grad-CAM
def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_class]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)

    return (
        heatmap.numpy(),
        int(predicted_class.numpy()),
        float(predictions[0][predicted_class].numpy())
    )


# Definicja modelu CNN
def model_cnn(num_classes=4, input_shape=(150, 150, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder - 3 bloki konwolucyjne, każdy blok ten sam wykorzystuje schemat
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs) # wykrywa proste wzorce: krawędzie, gradienty
    x = layers.BatchNormalization()(x) # stabilizuje uczenie, przyspiesza zbieżność
    x = layers.MaxPooling2D((2,2))(x) # zmniejsza wymiary przestrzenne: 150×150 → 75×75
    x = layers.Dropout(0.25)(x) # losowo wyłącza 25% neuronów → zapobiega overfittingowi

    x = layers.Conv2D(64,(3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128,(3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Klasyfikator - podejmowanie decyzji
    x = layers.GlobalAveragePooling2D()(x) # agreguje mapę cech 18×18×128 do wektora 128 wartości (zamiast Flatten() — mniej parametrów, mniejszy overfitting)
    x = layers.Dense(128, activation="relu")(x) # uczy się kombinacji wykrytych cech
    x = layers.Dropout(0.5)(x) # wyłącza 50% neuronów — mocniejsza regularyzacja
    outputs = layers.Dense(num_classes, activation="softmax")(x) # wyjście: 4 prawdopodobieństwa dla 4 klas guza

    return tf.keras.Model(inputs, outputs)


# Wykres accuracy
def plot_accuracy(histories, titles, suptitle):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1:
        axes = [axes]

    for ax, history, title in zip(axes, histories, titles):
        ax.plot(history.history["accuracy"], label="Train")
        ax.plot(history.history["val_accuracy"], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(suptitle, x=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Dystrybucja klas w zbiorze
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

# Utworzenie confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, title, cmap, ax):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=cmap, ax=ax)
    ax.set_title(title)

# Wyświetlenie top błędów
def plot_top_errors(errors, class_names, model_name="Model", n_show=8):
    errors_sorted = sorted(errors, key=lambda e: e["confidence"], reverse=True)
    n_show = min(n_show, len(errors_sorted))
    if n_show == 0:
        print(f"Brak błędów do wyświetlenia dla modelu {model_name}.")
        return

    n_cols = min(4, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    plt.figure(figsize=(4*n_cols, 4*n_rows))

    for i in range(n_show):
        err = errors_sorted[i]
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(err["image"])
        plt.title(
            f"Prawda: {class_names[err['true']]}\nPredykcja: {class_names[err['predicted']]} {err['confidence']:.0%}",
            color="red", fontsize=9
        )
        plt.axis("off")

    plt.suptitle(f"{model_name} - błędy z najwyższą pewnością", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Preprocessing dla CNN
def preprocess_CNN(data, path):
    image, label = data
    image = tf.image.central_crop(image, 0.8)
    image = tf.image.resize(image, (150, 150))
    image = image / 255.0
    image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return image, label


# Preprocessing dla EfficientNet
def preprocess_effNet(data, path):
    image, label = data
    image = tf.image.central_crop(image, 0.8)
    image = tf.image.resize(image, (150, 150))
    image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return image, label


# Wyświetlenie errorów w konsoli
def print_top_errors(errors, class_names, model_name, top_n=5):
    print(f"\n{'='*5} Najczęstsze pomyłki modelu {model_name} {'='*5}\n")
    print("Format: Prawda → Predykcja\n")
    for i, ((true, pred), count) in enumerate(
        Counter((class_names[e["true"]], class_names[e["predicted"]]) for e in errors).most_common(top_n), 1
    ):
        print(f"{i}. {true} → {pred}: {count} przypadków")
    print("\n")


# Usunięcie obrazów ze zbioru na podstawie listy ścieżek
def remove_paths_from_dataset(dataset, paths_to_remove):
    paths_to_remove = tf.constant(paths_to_remove)

    def keep_fn(data, path):
        is_equal = tf.reduce_any(tf.equal(paths_to_remove, path))
        return tf.logical_not(is_equal)

    return dataset.filter(keep_fn)

#Uruchomienie Grad-CAM
def run_gradcam(model, dataset, class_names, n=4, normalize_image=False, sigma=0):
    last_conv_layer = [l.name for l in model.layers if isinstance(l, layers.Conv2D)][-1]
    sample_x, sample_y = next(iter(dataset))

    for i in range(n):
        img = sample_x[i]
        true_class = int(np.argmax(sample_y[i].numpy()))
        img_batch = tf.expand_dims(img, axis=0)
        heatmap, pred_class, confidence = make_gradcam_heatmap(model, img_batch, last_conv_layer)
        display_img = img.numpy() / 255.0 if normalize_image else img.numpy()
        show_gradcam(display_img, heatmap, true_class, pred_class, confidence, class_names, sigma=sigma)

# Wyświetlenie wyników Grad-CAM
def show_gradcam(image, heatmap, true_class, pred_class, confidence, class_names, sigma = 10):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"Oryginał: {class_names[true_class]}", fontsize=10)
    axes[0].axis("off")

    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (image.shape[0], image.shape[1])).numpy().squeeze()
    if sigma > 0:
        heatmap_resized = gaussian_filter(heatmap_resized, sigma=sigma)
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Mapa Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.4)
    is_correct = true_class == pred_class
    axes[2].set_title(
        f"Predykcja: {class_names[pred_class]} ({confidence:.0%})",
        color="green" if is_correct else "red",
        fontsize=10
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# Wyświetlenie obrazów z listy ścieżek
def show_images(paths, rows=None, cols=None, figsize=(16, 8), update_path=False):
    n = len(paths)
    if rows is None and cols is None:
        cols = min(5, n)
        rows = math.ceil(n / cols)
    elif rows is None:
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)

    plt.figure(figsize=figsize)
    if update_path:
        paths = [path.replace('/kaggle/input/brain-tumor-mri-dataset', update_path) for path in paths]
    for i, path in enumerate(paths, 1):
        img = Image.open(path)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Wizualizacja predykcji modelu
def show_predictions(model, dataset, class_names, n=12, rows=3, cols=4, figsize=(10,6)):
    x, y = next(iter(dataset))
    pred = np.argmax(model.predict(x, verbose=0), axis=1)

    if len(y.shape) > 1:
        y = np.argmax(y.numpy(), axis=1)
    else:
        y = y.numpy()

    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x[i])
        plt.title(
            f"T:{class_names[y[i]]}\nP:{class_names[pred[i]]}",
            color="green" if y[i]==pred[i] else "red"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Podział datasetu na treningowy, walidacyjny i testowy
def split_train_val_test_ds(dataset, dataset_size, split_ratios={"val":0.15,"test":0.15,"train":0.7}, limit_ds=None):
    dataset = dataset.shuffle(buffer_size=dataset_size, seed=28)  # Zapewnienie powtarzalności
    train_dataset = dataset.take(int(split_ratios['train']*dataset_size))
    val_dataset = dataset.skip(int(split_ratios['train']*dataset_size)).take(int(split_ratios['val']*dataset_size))
    test_dataset = dataset.skip(int((split_ratios['train']+split_ratios['val'])*dataset_size))

    if limit_ds is not None:
        train_ratio = sum(1 for _ in train_dataset)*limit_ds
        test_ratio = sum(1 for _ in test_dataset)*limit_ds
        val_ratio = sum(1 for _ in val_dataset)*limit_ds
        train_dataset = train_dataset.take(train_ratio)
        test_dataset = test_dataset.take(test_ratio)
        val_dataset = val_dataset.take(val_ratio)

    return train_dataset, val_dataset, test_dataset