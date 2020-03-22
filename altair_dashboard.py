import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])

if st.checkbox('Show Data'):
   df

c = alt.Chart(df).mark_circle().encode(x='a', y='b', size='c',  
                                       color='c')
st.altair_chart(c, width=-1)