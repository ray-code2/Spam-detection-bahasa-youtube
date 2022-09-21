import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt
le = LabelEncoder()
from gsheetsdb import connect

# Create a connection object.
conn = connect()

@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

sheet_url = st.secrets["public_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

# Print results.
for row in rows:
    st.write(f"{row.name} has a :{row.pet}:")

pilih_menu = st.sidebar.selectbox("Navigasi" ,('Halaman Utama','Tentang Aplikasi'))

df = pd.read_csv("data.csv")
df['Value'] = le.fit_transform(df['Value'])
X = df['Komentar']
y = df['Value']

# st.write('Jumlah baris dan kolom', X.shape)
# st.write('Jumlah kelas: ',len(np.unique(y)))

tfidf = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7,ngram_range=(1,3))
text_tf = tfidf.fit_transform(X.astype('U'))
x_train,x_test,y_train,y_test = train_test_split(text_tf,y,test_size=0.2,random_state=42)
# st.markdown(y_test.shape)
text_classifier_linear = SVC(kernel='linear')
model = text_classifier_linear.fit(x_train, y_train)
predictions_svm = text_classifier_linear.predict(x_test)

def transform(komentar):
    komentar = komentar.lower() # mengubah menjadi huruf kecil
    komentar = komentar.strip() # menghilangkan suffix dan prefix yang kosong
    komentar = re.sub(r"\d+", "",komentar) # menghilangkan angka
    komentar = komentar.encode('ascii','replace').decode('ascii') #menghilangkan Non ascii
    komentar = komentar.translate(str.maketrans("","",string.punctuation)) #menghilangkan simbol
    komentar = re.sub("\s+", " ", komentar) #menghilangkan beberapa spasi kosong
    komentar = re.sub(r"\b[a-zA-Z]\b", " ",komentar) #menghilangkan single char
    komentar = komentar.replace('https\s+'," ")
    tf = tfidf.transform([komentar])
    return tf




if pilih_menu == 'Halaman Utama':
    st.title("Aplikasi Deteksi Komentar Spam Youtube dengan Metode SVM ")
    text_input = st.text_input('Input komentar','Input disini')
    input_button = st.button('Prediksi')
    if text_input == '':
        st.warning('Belum input komentar!')
    else:
      if input_button:
        tf = transform(text_input)
        pred_text = model.predict(tf)
        pred_text = le.inverse_transform(pred_text)
        text = ' '.join(pred_text)
        st.write('Komentar',text_input,'termasuk kategori: ', f'**{text}**' )
        akurasi = round(accuracy_score(y_test,predictions_svm)*100,2)
        st.markdown(f'''Hasil akurasi SVM memprediksi input: **{akurasi}%** ''' )

   
else:   
    st.write("ini halaman Tentang Aplikasi")
   


   


