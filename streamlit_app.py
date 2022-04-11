from tkinter import Button
import pandas
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import streamlit as st

st.set_page_config(layout="centered", page_icon="🎓", page_title="Excel File Generator")

# sidebar
with st.sidebar:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file")
    if uploaded_file is not None:
        
        st.sidebar.success("🎉 Your diploma was generated!")
    else:

        st.sidebar.error("No file uploaded yet!")



# page
st.header("EXCEL FILE GENERATOR WEB APP")
st.text("With the help of this app you can convert your CSV file to Excel.")

if uploaded_file is not None:

    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    st.write("Uploaded CSV File")
    st.write(df)

    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    df_xlsx = to_excel(df)
    st.download_button(label='📥 Download Excel File', data=df_xlsx , file_name= 'workbook.xlsx')

else:
    st.info("Awaiting for file")