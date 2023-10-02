


import streamlit as st

st.set_page_config(
    page_title="Sangkak Challenge: Elvis MBONING",
    page_icon="👋",
)

st.title('Sangkak Challenge Session septembre 2023')


st.sidebar.success("Sangkak Challenge Session septembre 2023")
st.sidebar.success("Build by ELvis MBONING")


st.markdown("""
    # Introduction
            
    Cette section résume l'ensemble des expérimentations réalisées
    dans le cadre du challenge de cette section.
            
    - Auteur: Elvis MBONING
    - Thématique: POS Tagging
    - Challenge session: septembre 2023

    # Expérimentations: propositions techniques
    
    We try to implement new methods which can potentialy improved POS task
    in low african resource languages.

    We propose a rule-based approach call **Position to position entity augmentation** 
    to normalize and augment lowest training data for CRF model. 
    Our work is based on this paper 
    [Xiang Dai and Heike Adel, 2020](https://aclanthology.org/2020.coling-main.343.pdf).
    
""")


