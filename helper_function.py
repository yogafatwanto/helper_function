import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import os
import itertools
from scipy.special import softmax
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, f1_score, classification_report, roc_auc_score, auc, roc_curve, confusion_matrix

def get_auc(model, model_name, X_train, y_train, X_test, y_test):
    """
    Menghitung nilai Area Under Curve (AUC) menggunakan model yang diberikan.
    
    Args:
        model: Model yang akan digunakan untuk menghitung AUC.
        model_name (str): Nama model.
        X_train: Data fitur pelatihan.
        y_train: Data target pelatihan.
        X_test: Data fitur pengujian.
        y_test: Data target pengujian.
        
    Returns:
        float: Nilai AUC.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr, tpr, label=f"{model_name}, AUC="+str(auc))
    plt.legend()
    return auc

def divided_box_hist_plot(series, y):
    """
    Menampilkan boxplot dan histogram terbagi berdasarkan label target.
    
    Args:
        series: Data series yang akan divisualisasikan.
        y: Label target yang sesuai dengan series.
    """
    labels = y.unique()
    gs_kw = dict(width_ratios=[.5, .5], height_ratios=[.15, .85])
    f, ax = plt.subplots(2, 2, sharex=True, gridspec_kw=gs_kw)
    sns.boxplot(x=series[y==labels[0]], color="red", ax=ax[0][0])
    sns.boxplot(x=series[y==labels[1]], color="blue", ax=ax[0][1])
    ax[0][0].set(xlabel="")
    ax[0][1].set(xlabel="")
    sns.distplot(x=series[y==labels[0]].dropna(), color="red", ax=ax[1][0], kde=False)
    sns.distplot(x=series[y==labels[1]].dropna(), color="blue", ax=ax[1][1], kde=False)
    ax[1][0].set(xlabel=f"{series.name} ({labels[0]})")
    ax[1][1].set(xlabel=f"{series.name} ({labels[1]})")
    plt.show()

# Membuat fugsi untuk plot roc curve
def plot_roc_curve(fprs, tprs):
    """
    Menampilkan kurva ROC berdasarkan nilai true positive rates dan false positive rates.
    
    Args:
        fprs: List nilai false positive rates.
        tprs: List nilai true positive rates.
        
    Returns:
        tuple: Objek matplotlib Figure dan Axes.
    """
    # Inisialisasi list + sumbu plot..
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC untuk setiap K-Fold + hitung AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    
    # Plot mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    
    # Plot standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Fine tune plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)

def compute_roc_auc(index):
    """
    Menghitung nilai False Positive Rate (FPR), True Positive Rate (TPR), dan Area Under Curve (AUC) berdasarkan indeks data.
    
    Args:
        index: Indeks data yang akan digunakan untuk menghitung ROC AUC.
        
    Returns:
        tuple: FPR, TPR, dan nilai AUC.
    """
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def get_information_values(data, target, bins=10, show_woe=False):
    """
    Menghitung nilai Information Value (IV) dan Weight of Evidence (WoE) untuk setiap variabel independen dalam data.
    
    Args:
        data: Dataframe yang berisi variabel independen dan target.
        target: Nama kolom target dalam data.
        bins (int): Jumlah interval/bin yang digunakan saat menghitung WoE dan IV untuk variabel numerik.
        show_woe (bool): Menampilkan tabel WoE untuk setiap variabel jika bernilai True.
        
    Returns:
        tuple: Dataframe IV dan dataframe WoE.
    """
    # Buat dataframe kosong
    data_iv, data_woe = pd.DataFrame(), pd.DataFrame()
    
    # Extract kolom
    cols = data.columns
    
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        data_iv = pd.concat([data_iv, temp], axis=0)
        data_woe = pd.concat([data_woe, d], axis=0)

        # Menampilkan WOE Table
        if show_woe == True:
            print(d)

    return data_iv, data_woe


from scipy.special import softmax

def print_feature_importances_shap_values(shap_values, features):
    '''
    Mencetak nilai feature importances berdasarkan SHAP values dalam urutan yang terurut.
    
    Args:
        shap_values: Nilai-nilai SHAP yang dihitung dari objek shap.Explainer.
        features: Nama-nama fitur, sesuai dengan urutan yang diberikan ke objek explainer.
    '''
    # Menghitung nilai feature importance (mean absolute SHAP value) untuk setiap fitur
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Menghitung versi yang ternormalisasi
    importances_norm = softmax(importances)
    # Mengorganisir nilai importances dan kolom dalam sebuah kamus
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Mengurutkan kamus
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
    feature_importances_norm = {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
    # Mencetak nilai feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        
        
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Membaca gambar dari file `filename`, mengubahnya menjadi tensor, dan mengubah ukurannya menjadi (224, 224, 3).

    Parameters
    ----------
    filename (str): Nama file gambar target.
    img_shape (int): Ukuran untuk meresize gambar target, default 224.
    scale (bool): Menentukan apakah nilai piksel akan diubah skala menjadi rentang (0, 1), default True.
    
    Returns
    -------
    tf.Tensor: Gambar yang telah diubah menjadi tensor dengan ukuran (224, 224, 3).

    """
    # Membaca gambar
    img = tf.io.read_file(filename)
    # Mendecode gambar menjadi tensor
    img = tf.image.decode_jpeg(img)
    # Meresize gambar
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Mengubah skala gambar (nilai antara 0 dan 1)
        return img/255.
    else:
        return img


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """
    Membuat confusion matrix berlabel yang membandingkan prediksi dan label ground truth.

    Jika `classes` diberikan, maka confusion matrix akan diberi label. Jika tidak, maka akan digunakan nilai kelas dalam bentuk bilangan bulat.

    Args:
        y_true: Array dari label kebenaran (harus memiliki bentuk yang sama dengan `y_pred`).
        y_pred: Array dari label prediksi (harus memiliki bentuk yang sama dengan `y_true`).
        classes: Array dari label kelas (misalnya, dalam bentuk string). Jika `None`, maka label bilangan bulat akan digunakan.
        figsize: Ukuran output figure (default=(10, 10)).
        text_size: Ukuran teks dalam output figure (default=15).
        norm: Menormalisasi nilai atau tidak (default=False).
        savefig: Menyimpan confusion matrix ke file (default=False).

    Returns:
        matplotlib.pyplot.figure: Confusion matrix berlabel yang membandingkan `y_true` dan `y_pred`.

    Example usage:
        make_confusion_matrix(y_true=test_labels,  # label kebenaran dari data uji
                              y_pred=y_preds,  # label prediksi
                              classes=class_names,  # array dari label kelas
                              figsize=(15, 15),
                              text_size=10)
    """
    # Membuat confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # melakukan normalisasi
    n_classes = cm.shape[0]  # mendapatkan jumlah kelas yang ada

    # Plotting figure dan mengatur tampilan
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # warna akan mewakili tingkat 'kebenaran' dari suatu kelas, semakin gelap semakin baik
    fig.colorbar(cax)

    # Apakah terdapat daftar kelas?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Memberi label pada sumbu-sumbu
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # membuat slot sumbu yang cukup untuk setiap kelas
           yticks=np.arange(n_classes),
           xticklabels=labels,  # sumbu akan diberi label dengan nama kelas (jika ada) atau bilangan bulat
           yticklabels=labels)
    
    # Menampilkan label sumbu x di bagian bawah
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Menentukan ambang batas untuk warna yang berbeda
    threshold = (cm.max() + cm.min()) / 2.

    # Menampilkan teks pada setiap sel
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > threshold else "black",
                      size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > threshold else "black",
                      size=text_size)

    # Menyimpan gambar ke direktori kerja saat ini
    if savefig:
        fig.savefig("confusion_matrix.png")


  
def pred_and_plot(model, filename, class_names):
    """
    Mengimpor gambar yang terletak di `filename`, membuat prediksi dengan menggunakan
    model yang telah dilatih, dan memplot gambar beserta kelas prediksi sebagai judul.

    Args:
        model: Model yang telah dilatih.
        filename: Nama file dari gambar target.
        class_names: Daftar nama kelas.

    Returns:
        None. Memplot gambar beserta kelas prediksi sebagai judul.
    """
    # Mengimpor gambar target dan memprosesnya
    img = load_and_prep_image(filename)

    # Melakukan prediksi
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Mendapatkan kelas prediksi
    if len(pred[0]) > 1:  # Memeriksa untuk multi-kelas
        pred_class = class_names[pred.argmax()]  # Jika lebih dari satu output, ambil yang terbesar
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # Jika hanya satu output, dibulatkan

    # Memplot gambar dan kelas prediksi
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Membuat TensorBoard callback instance untuk menyimpan file log.

    Menyimpan file log dengan path:
        "dir_name/experiment_name/current_datetime/"

    Args:
        dir_name: Direktori target untuk menyimpan file log TensorBoard.
        experiment_name: Nama direktori eksperimen (misalnya, efficientnet_model_1).

    Returns:
        tf.keras.callbacks.TensorBoard: Callback TensorBoard untuk menyimpan file log.
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def plot_loss_curves(history):
    """
    Memplot kurva loss terpisah untuk metrik pelatihan dan validasi.

    Args:
        history: Objek History dari model TensorFlow (lihat: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History).

    Returns:
        None. Memplot kurva loss dan akurasi.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Memplot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Memplot akurasi
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Membandingkan dua objek History model TensorFlow.

    Args:
        original_history: Objek History dari model asli (sebelum new_history).
        new_history: Objek History dari pelatihan model yang dilanjutkan (setelah original_history).
        initial_epochs: Jumlah epoch pada original_history (plot new_history dimulai dari sini).

    Returns:
        None. Memplot kurva akurasi dan loss pelatihan serta validasi.
    """
    # Mendapatkan pengukuran history dari model asli
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Menggabungkan history asli dengan new_history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Memplot kurva
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # Menyusun ulang plot sekitar epoch
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # Menyusun ulang plot sekitar epoch
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def unzip_data(filename):
    """
    Mengekstrak file zip (filename) ke dalam direktori kerja saat ini.

    Args:
        filename: Nama file dari folder zip target yang akan diekstrak.

    Returns:
        None. Mengekstrak file zip ke dalam direktori kerja saat ini.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


def count_files(directory):
    """
    Menghitung jumlah file (gambar) dalam setiap subdirektori di dalam direktori klasifikasi gambar.

    Args:
        directory: Direktori yang akan diperiksa.

    Returns:
        dict: Dictionary dengan nama subdirektori sebagai kunci dan jumlah file dalam subdirektori sebagai nilai.
    """
    file_counts = {}
    for root, dirs, files in os.walk(directory):
        if dirs:
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                file_counts[dir_name] = len(os.listdir(dir_path))
    return file_counts

def walk_through_dir(dir_path):
    """
    Melakukan penelusuran pada dir_path dan mengembalikan isinya.

    Args:
        dir_path (str): direktori target.
        
    Prints:
        Jumlah subdirektori di dir_path.
        Jumlah gambar (file) dalam setiap subdirektori.
        Nama setiap subdirektori.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def calculate_results(y_true, y_pred):
    """
    Menghitung akurasi, presisi, recall, dan skor f1 dari model klasifikasi biner.

    Args:
        y_true: label sebenarnya dalam bentuk array 1 dimensi.
        y_pred: label yang diprediksi dalam bentuk array 1 dimensi.

    Returns:
        Dictionary yang berisi akurasi, presisi, recall, dan skor f1.
    """
    # Hitung akurasi model
    model_accuracy = accuracy_score(y_true, y_pred)
    # Hitung presisi, recall, dan skor f1 model menggunakan "rata-rata tertimbang"
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1
    }
    return model_results


def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Membuat plot timesteps (serangkaian titik dalam waktu) terhadap nilai-nilai (serangkaian nilai sepanjang timesteps).
  
    Parameters
    ----------
    timesteps : array dari timesteps
    values : array dari nilai-nilai sepanjang waktu
    format : gaya plot, default "."
    start : posisi mulai plot (mengatur nilai akan mengindeks dari awal timesteps & values)
    end : posisi akhir plot (mengatur nilai akan mengindeks dari akhir timesteps & values)
    label : label yang ditampilkan pada plot nilai-nilai
    """
    # Plot rangkaian data
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14) # memperbesar label
    plt.grid(True)


def mean_absolute_scaled_error(y_true, y_pred):
    """
    Mengimplementasikan MASE (asumsi tidak ada musiman pada data).
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Temukan MAE dari ramalan naif (tanpa musiman)
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # musiman kita adalah 1 hari (maka pergeseran 1 hari)

    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    """
    Menghitung berbagai metrik untuk evaluasi hasil prediksi.

    Args:
        y_true: label sebenarnya dalam bentuk array.
        y_pred: label yang diprediksi dalam bentuk array.

    Returns:
        Dictionary yang berisi nilai mae, mse, rmse, mape, dan mase.
    """
    # Pastikan tipe data float32 (untuk perhitungan metrik)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Hitung berbagai metrik
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # memberikan penekanan pada nilai-nilai ekstrem (semua error dipangkatkan dua)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}
