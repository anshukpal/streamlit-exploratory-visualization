import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

def main():
    df = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

    if page == "Homepage":
        st.header("This is your data explorer.")
        st.write("Please select a page on the left.")
        st.write(df)
    elif page == "Exploration":
        st.title("Data Exploration")
        x_axis = st.selectbox("Choose a variable for the x-axis", df.columns)
        y_axis = st.selectbox("Choose a variable for the y-axis", df.columns)
        visualize_data(df, x_axis, y_axis)

@st.cache
def load_data():
   data = pd.read_csv('ProcterGambleStock.csv')
   return data

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis
    ).interactive()

    st.write(graph)

if __name__ == "__main__":
    main()