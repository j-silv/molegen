import streamlit as st

home_page = st.Page("pages/home.py", title="Home", default=True)
data_page = st.Page("pages/data.py", title="ZINC dataset exploration")
pg = st.navigation([home_page, data_page], position='top')

st.set_page_config(page_title="MoleGen")

pg.run()