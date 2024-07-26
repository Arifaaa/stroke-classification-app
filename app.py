import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import joblib
import time

from streamlit_option_menu import option_menu
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
# prepocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE



@st.cache_data()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)

df = pd.read_csv('data.csv')

data = pd.DataFrame(df)

# Identifikasi fitur kategorikal dan kontinu
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
continuous_features = ['age', 'avg_glucose_level', 'bmi']
continuous_features1 = ['age', 'avg_glucose_level']

df_encode = pd.read_excel('setelah-encoding.xlsx')
df_final = pd.read_excel('hasil imputasi.xlsx')

df_outlier_handled = pd.read_excel('minmax.xlsx')
df_dropna_outlier_handled = pd.read_excel('drop_minmax.xlsx')

# Define the Dependent and independent variable.
X_drop = df_dropna_outlier_handled.drop(['stroke'], axis=1)
y_drop = df_dropna_outlier_handled['stroke']

# Define the Dependent and independent variable.
X = df_outlier_handled.drop(['stroke'], axis=1)
y = df_outlier_handled['stroke']

# SMOTE
smotek3 = SMOTE(k_neighbors=3, random_state=42)
smotek5 = SMOTE(k_neighbors=5, random_state=42)
smotek7 = SMOTE(k_neighbors=7, random_state=42)

X_resampled_k3, y_resampled_k3 = smotek3.fit_resample(X, y)
X_resampled_k5, y_resampled_k5 = smotek5.fit_resample(X, y)
X_resampled_k7, y_resampled_k7 = smotek7.fit_resample(X, y)

cat_test = joblib.load('model/cat_nbk5_model.pkl')
gauss_test = joblib.load('model/gauss_nbk5_model.pkl')
label_encoder = joblib.load('model/LE.pkl')
scaler_con = joblib.load('model/scaler_con.pkl')
categorical_features = joblib.load('model/cat.pkl')
continuous_features = joblib.load('model/con.pkl')

def preprocess_new_data(new_data):
    new_data = new_data[categorical_features + continuous_features]

    for feature in categorical_features:
        new_data[feature] = label_encoder[feature].transform(new_data[feature])

    new_data[continuous_features] = scaler_con.transform(new_data[continuous_features])
    return new_data

def predict_new_data(new_data):
    new_data = preprocess_new_data(new_data)
    # Separate the new data into categorical and continuous features
    new_data_cat = new_data[categorical_features]
    new_data_cont = new_data[continuous_features]

    print(new_data_cat)
    # Predict probabilities
    cat_probs = cat_test.predict_proba(new_data_cat)
    gauss_probs = gauss_test.predict_proba(new_data_cont)

    # Combine probabilities
    combined_probs = cat_probs * gauss_probs

    # Make final predictions
    combined_preds = np.argmax(combined_probs, axis=1)
    return combined_preds

with st.sidebar:
    selected = option_menu('',['Home', 'Research', 'Dataset', 'Preprocessing', 'Modelling', 'Implementation'], default_index=0)

if (selected == 'Home'):
    st.markdown("<h1 style='text-align: center; '>Implementasi Metode Smote Pada Klasifikasi Penyakit Stroke dengan Algoritma Naive Bayes</h1>", unsafe_allow_html=True)
    url_logo = 'utm.png'
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(url_logo, use_column_width='auto')
    with col3:
        st.write(' ')

    st.markdown("<h5 style='text-align: center; '>Diajukan Untuk Memenuhi Persyaratan Penyelesaian Studi Strata Satu (S1) dan Memperoleh Gelar Sarjana Komputer (S.Kom) di Universitas Trunojoyo Madura</h5>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>Disusun Oleh:</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>Arifatul Maghfiroh</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>NPM:200411100201</h6>", unsafe_allow_html=True)

    dp = [
        ("Dosen Pembimbing 1", "Achmad Jauhari, S.T., M.Kom"),
        ("Dosen Pembimbing 2", "Fifin Ayu Mufarroha, S.Kom., M.Kom")
    ]
    dp_table = pd.DataFrame(dp, columns=["Role", "Name"])
    st.table(dp_table)

    dpn = [
        ("Dosen Penguji 1", "Mulaab, S.Si., M.Kom"),
        ("Dosen Penguji 2", "Andharini Dwi Cahyani, S.Kom., M.Kom"),
        ("Dosen Penguji 3", "Arik Kurniawati, S.Kom., M.T")
    ]
    dpn_table = pd.DataFrame(dpn, columns=["Role", "Name"])
    st.table(dpn_table)


if (selected == 'Research'):
    st.markdown("<h1 style='text-align: center; '>Implementasi Metode Smote Pada Klasifikasi Penyakit Stroke dengan Algoritma Naive Bayes</h1>", unsafe_allow_html=True)
    st.subheader("""Latar Belakang""")
    st.write("""Stroke adalah penyebab umum kematian dan kecacatan di seluruh dunia, yang terjadi ketika pembuluh darah di otak tersumbat atau pecah, 
    yang mengakibatkan hilangnya suplai darah dan oksigen ke otak. Dokter spesialis saraf menggunakan analisis gejala dan pemeriksaan neurologis untuk mendiagnosis stroke, 
    tetapi hal ini bisa menjadi tantangan dan membutuhkan ketelitian. Untuk membantu pengobatan dini dan mengurangi angka kematian, prediksi stroke yang akurat sangat penting. 
    Penelitian ini mengusulkan penggunaan pembelajaran mesin untuk mendiagnosis stroke dengan lebih akurat menggunakan data yang tidak seimbang. 
    Namun, data yang memiliki ketidakseimbangan kelas dapat memberikan suatu kesalahan pada penentuan kelas sehingga performa yang dihasilkan menurun. 
    Untuk mengatasi hal ini, penelitian ini menggunakan teknik oversampling yaitu Synthetic Minority Oversampling Technique (SMOTE) untuk menyeimbangkan data, bersama dengan metode klasifikasi Naïve Bayes. 
    Tujuannya adalah untuk mengembangkan model pembelajaran mesin yang dapat secara akurat memprediksi apakah seseorang menderita stroke atau tidak, agar dapat memudahkan proses diagnosis stroke bagi para profesional medis.
     """)
    st.subheader("""Metode Usulan""")
    st.write("""Penelitian ini dilakukan untuk menangani imbalance pada data penyakit stroke menggunakan metode SMOTE yang selanjutnya dilakukan proses klasifikasi menggunakan algoritma Naïve Bayes. 
             Pengujian performa klasifikasi dilakukan dengan melihat nilai statistik dari accuracy, precision, recall, dan f1-score. """)
    st.subheader("""Tujuan Penelitian""")
    st.write("""Tujuan dari penelitana ini adalah untuk mengetahui pengaruh yang dihasilkan dari proses penanganan imbalanced data pada data yang tidak seimbang menggunakan metode SMOTE dalam menghasilkan klasifikasi penyakit stroke. """)
if (selected == 'Dataset'):
    st.title("Dataset")
    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with dataset:
        dataset = pd.read_csv('https://raw.githubusercontent.com/Arifaaa/dataset/main/healthcare-dataset-stroke-data.csv')
        # Optional: Display dataset
        if st.checkbox("Show Dataset"):
            st.write(dataset)

        selected_feature = st.selectbox("##### Pilih fitur untuk di visualisasi", dataset.columns)

        # Main content
        st.subheader("Distribusi Data")
        st.write(f"##### Fitur: {selected_feature}")

        # Plot distribution using seaborn
        fig, ax = plt.subplots()
        sns.histplot(dataset[selected_feature], kde=True, ax=ax)
        ax.set_title(f'Distribution of {selected_feature}')
        st.pyplot(fig)

    with ket:
        st.write(
            "Data yang digunakan diperoleh dari repositori Kaggle yang terdiri dari catatan kesehatan yang dikumpulkan dari berbagai rumah sakit di Bangladesh oleh tim peneliti untuk tujuan akademis. Data ini dapat diakses secara publik melalui https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
        st.download_button(
            label="Download data",
            data='dataset/data.xlsx',
            file_name='data.xlsx',
            mime='text/xlsx',
        )
        st.write("""
            Keterangan Dataset :
        """)
        ket_data = pd.read_csv('ket_data.csv')
        st.dataframe(ket_data)

if (selected == 'Preprocessing'):
    st.title("Preprocessing Data")
    selection, labelEnc, KNNimp, outlier, minmaxScaler, smote = st.tabs(["Data Selection", "Label Encoder", "KNN Imputation", "Penanganan Outlier", "MinMaxScaler", "SMOTE"])
    with selection:
        dataset = pd.read_csv('https://raw.githubusercontent.com/Arifaaa/dataset/main/healthcare-dataset-stroke-data.csv')
        st.write("#### Distribusi data pada atribut 'work_type' dan 'gender'")
        value_counts = dataset[['work_type', 'gender']].value_counts().reset_index()
        value_counts.columns = ['work_type', 'gender', 'count']
        st.dataframe(value_counts, width=500, height=430)
        st.write("#### Menghapus atribut id dan kategori 'other' pada atribut gender")
        data = pd.read_csv('data.csv')
        st.dataframe(data)
    with labelEnc:
        st.write("""Encoding data :""")
        enc = pd.read_csv('enc.csv')
        st.dataframe(enc, width=500)
        st.write("""Hasil Encoding data :""")
        st.dataframe(df_encode)
    with KNNimp:
        st.write("""Informasi missing value :""")
        mis = df.isnull().sum().reset_index()
        mis.columns = ['Fitur', 'Jumlah Missing Values']
        st.dataframe(mis, width=400)
        st.write("""Imputasi data menggunakan KNN Imputation dengan menghitung jarak tetangga terdekat menggunakan Euclidean Distance untuk data numerik dan Hamming Distance untuk data kategorik""")
        st.write("Rumus Euclidean Distance:")
        st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
        st.write("Rumus Hamming Distance:")
        st.latex(r"d_H(x, y) = \sum_{i=1}^{n} \mathbb(x_i \neq y_i)")
        st.write("""Hasil Imputasi :""")
        st.dataframe(df_final)
    with outlier:
        st.write("Penanganan Outlier")
        st.write("""Visualisasi data sebelum penanganan outlier :""")
        plt.figure(figsize=(12,9))
        for i,col in enumerate(df.select_dtypes(float).columns):
            plt.subplot(3,1,i+1)
            sns.boxplot(data=df,x=col,palette =sns.color_palette("Set2"))
        st.pyplot(plt)
        st.write("""Visualisasi data setelah penanganan outlier :""")
        plt.figure(figsize=(12,9))
        for i,col in enumerate(df.select_dtypes(float).columns[1:]):
            plt.subplot(2,1,i+1)
            sns.boxplot(data=df_outlier_handled,x=col,palette =sns.color_palette("Set2"))
        st.pyplot(plt)
    with minmaxScaler:
        st.write("""Normalisasi data menggunakan Min-Max Normalization""")
        st.write("Rumus Min-Max :")
        st.latex(r"x_{\text{normalized}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}")
        st.write("""Hasil Normalisasi :""")
        st.dataframe(df_outlier_handled)
    with smote:
        st.write("""### Visualisasi data sebelum oversampling :""")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=X_resampled_k5, x="age", y="avg_glucose_level", hue=y, palette="Set2")
        plt.xlabel("age")
        plt.ylabel("avg_glucose_level")
        plt.title("Persebaran Data Target")
        st.pyplot(plt)
        if st.checkbox("Show Dataset"):
            st.write(X_resampled_k3)
        fig, ax = plt.subplots()
        sns.histplot(y_resampled_k3, kde=True, ax=ax)
        ax.set_title(f'Distribution of stroke')
        st.pyplot(fig)
        

if (selected == 'Modelling'):
    progress()
    with st.form("Modelling"):
        st.subheader('Modelling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        nb = st.checkbox('Naive Bayes')
        nb_knni = st.checkbox('Naive Bayes + KNN Imputation')
        nb_k3 = st.checkbox('Naive Bayes + KNN Imputation + SMOTE(k=3)')
        nb_k5 = st.checkbox('Naive Bayes + KNN Imputation + SMOTE(k=5)')
        nb_k7 = st.checkbox('Naive Bayes + KNN Imputation + SMOTE(k=7)')
        submitted = st.form_submit_button("Submit")

        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_drop, y_drop, test_size=0.20, random_state=42)
        X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.20, random_state=42)
        X_train_k3, X_test_k3, y_train_k3, y_test_k3 = train_test_split(X_resampled_k3, y_resampled_k3, test_size=0.20, random_state=42)
        X_train_k5, X_test_k5, y_train_k5, y_test_k5 = train_test_split(X_resampled_k5, y_resampled_k5, test_size=0.20, random_state=42)
        X_train_k7, X_test_k7, y_train_k7, y_test_k7 = train_test_split(X_resampled_k7, y_resampled_k7, test_size=0.20, random_state=42)


        cat_nb = joblib.load('model/cat_nb_model.pkl')
        gauss_nb = joblib.load('model/gauss_nb_model.pkl')
        X_train_cat_nb = X_train_nb[categorical_features]
        X_train_cont_nb = X_train_nb[continuous_features1]
        X_test_cat_nb = X_test_nb[categorical_features]
        X_test_cont_nb = X_test_nb[continuous_features1]
        cat_probs = cat_nb.predict_proba(X_test_cat_nb)
        gauss_probs = gauss_nb.predict_proba(X_test_cont_nb)
        combined_probs = cat_probs * gauss_probs
        combined_preds = np.argmax(combined_probs, axis=1)
        accuracy = accuracy_score(y_test_nb, combined_preds)
        report = classification_report(y_test_nb, combined_preds)
        cm = confusion_matrix(y_test_nb,combined_preds)


        cat_knn = joblib.load('model/cat_knn_model.pkl')
        gauss_knn = joblib.load('model/gauss_knn_model.pkl')
        X_train_cat_knn = X_train_knn[categorical_features]
        X_train_cont_knn = X_train_knn[continuous_features]
        X_test_cat_knn = X_test_knn[categorical_features]
        X_test_cont_knn = X_test_knn[continuous_features]
        cat_probs_knn = cat_knn.predict_proba(X_test_cat_knn)
        gauss_probs_knn = gauss_knn.predict_proba(X_test_cont_knn)
        combined_probs_knn = cat_probs_knn * gauss_probs_knn
        combined_preds_knn = np.argmax(combined_probs_knn, axis=1)
        accuracy_knn = accuracy_score(y_test_knn, combined_preds_knn)
        report_knn = classification_report(y_test_knn, combined_preds_knn)
        cm_knn = confusion_matrix(y_test_knn,combined_preds_knn)

        cat_k3 = joblib.load('model/cat_nbk3_model.pkl')
        gauss_k3 = joblib.load('model/gauss_nbk3_model.pkl')
        X_train_cat_k3 = X_train_k3[categorical_features]
        X_train_cont_k3 = X_train_k3[continuous_features]
        X_test_cat_k3 = X_test_k3[categorical_features]
        X_test_cont_k3 = X_test_k3[continuous_features]
        cat_probs_k3 = cat_k3.predict_proba(X_test_cat_k3)
        gauss_probs_k3 = gauss_k3.predict_proba(X_test_cont_k3)
        combined_probs_k3 = cat_probs_k3 * gauss_probs_k3
        combined_preds_k3 = np.argmax(combined_probs_k3, axis=1)
        accuracy_k3 = accuracy_score(y_test_k3, combined_preds_k3)
        report_k3 = classification_report(y_test_k3, combined_preds_k3)
        cm3 = confusion_matrix(y_test_k3, combined_preds_k3)

        cat_k5 = joblib.load('model/cat_nbk5_model.pkl')
        gauss_k5 = joblib.load('model/gauss_nbk5_model.pkl')
        X_train_cat_k5 = X_train_k5[categorical_features]
        X_train_cont_k5 = X_train_k5[continuous_features]
        X_test_cat_k5 = X_test_k5[categorical_features]
        X_test_cont_k5 = X_test_k5[continuous_features]
        cat_probs_k5 = cat_k5.predict_proba(X_test_cat_k5)
        gauss_probs_k5 = gauss_k5.predict_proba(X_test_cont_k5)
        combined_probs_k5 = cat_probs_k5 * gauss_probs_k5
        combined_preds_k5 = np.argmax(combined_probs_k5, axis=1)
        accuracy_k5 = accuracy_score(y_test_k5, combined_preds_k5)
        report_k5 = classification_report(y_test_k5, combined_preds_k5)
        cm5 = confusion_matrix(y_test_k5, combined_preds_k5)

        cat_k7 = joblib.load('model/cat_nbk7_model.pkl')
        gauss_k7 = joblib.load('model/gauss_nbk7_model.pkl')
        X_train_cat_k7 = X_train_k7[categorical_features]
        X_train_cont_k7 = X_train_k7[continuous_features]
        X_test_cat_k7 = X_test_k7[categorical_features]
        X_test_cont_k7 = X_test_k7[continuous_features]
        cat_probs_k7 = cat_k7.predict_proba(X_test_cat_k7)
        gauss_probs_k7 = gauss_k7.predict_proba(X_test_cont_k7)
        combined_probs_k7 = cat_probs_k7 * gauss_probs_k7
        combined_preds_k7 = np.argmax(combined_probs_k7, axis=1)
        accuracy_k7 = accuracy_score(y_test_k7, combined_preds_k7)
        report_k7 = classification_report(y_test_k7, combined_preds_k7)
        cm7 = confusion_matrix(y_test_k7, combined_preds_k7)


        if submitted :
            if nb :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(accuracy))
                st.write('Classification report: ')
                st.text(report)
            if nb_knni :
                st.write('Model Naive Bayes KNNI accuracy score: {0:0.2f}'. format(accuracy_knn))
                st.text(report_knn)
            if nb_k3 :
                st.write('Model Naive Bayes KNNI SMOTE k=3  accuracy score: {0:0.2f}'. format(accuracy_k3))
                st.text(report_k3)
            if nb_k5 :
                st.write('Model Naive Bayes KNNI SMOTE k=5 accuracy score: {0:0.2f}'. format(accuracy_k5))
                st.text(report_k5)
            if nb_k7 :
                st.write('Model Naive Bayes KNNI SMOTE k=7 accuracy score: {0:0.2f}'. format(accuracy_k7))
                st.text(report_k7)
        
        cmatrix = st.form_submit_button("COnfusion Matrix semua model")
        if cmatrix:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))

            # Plot pertama
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0])
            axes[0, 0].set_title('NB')

            # Plot kedua
            sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 1])
            axes[0, 1].set_title('NB + KNNI')

            # Plot ketiga
            sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 2])
            axes[0, 2].set_title('NB + KNNI + SMOTE (k=3)')

            # Plot keempat
            sns.heatmap(cm5, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1, 0])
            axes[1, 0].set_title('NB + KNNI + SMOTE (k=5)')

            # Plot kelima
            sns.heatmap(cm7, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1, 1])
            axes[1, 1].set_title('NB + KNNI + SMOTE (k=7)')

            # Hapus subplot yang tidak digunakan
            fig.delaxes(axes[1, 2])

            # Atur label dan judul keseluruhan
            plt.xlabel('Prediction')
            plt.ylabel('Actual')
            plt.suptitle('Confusion Matrix untuk semua model')
            st.pyplot(fig)

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [accuracy, accuracy_knn, accuracy_k3, accuracy_k5, accuracy_k7],
                'Model' : ['Naive Bayes', 'NB+KNNI', 'NB+KNNI+SMOTE3', 'NB+KNNI+SMOTE5', 'NB+KNNI+SMOTE7'],
            })
            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.Y("Akurasi"),
                    alt.X("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)


if (selected == "Implementation"):
     with st.form("my_form"):
        st.subheader("Implementation")
        gender = st.selectbox('Gender', options=["Male", "Female"])
        age = st.number_input('Age', min_value=0, max_value=100, value=50)
        hypertension = st.selectbox('Hypertension', options=[0, 1])
        heart_disease = st.selectbox('Heart Disease', options=[0, 1])
        ever_married = st.selectbox('Ever Married', options=["Yes", "No"])
        work_type = st.selectbox('Work Type', options=['Private', 'Self-employed', 'Govt job', 'children', 'Never worked'])
        residence_type = st.selectbox('Residence Type', options=['Urban', 'Rural'])
        avg_glucose_level = st.number_input('Avg Glucose Level', min_value=0.0, value=100.0)
        bmi = st.number_input('BMI', min_value=0.0, value=25.0)
        smoking_status = st.selectbox('Smoking Status', options=['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
        
        prediksi = st.form_submit_button("Predict")
        if prediksi:
            input = {
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            }
            input_data = pd.DataFrame(input)

            prediction = predict_new_data(input_data)

            st.subheader('Prediction Results')
            # st.write('Using the', model, 'model')
            if (prediction[0] == 0):
                st.success('Tidak stroke')
            else:
                st.error('Stroke')
