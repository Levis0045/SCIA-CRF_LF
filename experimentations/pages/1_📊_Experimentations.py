import streamlit as st
from pathlib import Path

st.title("Data source exploration")

st.markdown("""
    Cette section résume l'ensemble des expérimentations réalisées
    dans le cadre du chalenge de cette section.

""")
            
col1, col2 = st.columns(2)


data_path = "./data_source/masakhane-pos/data"
data_path_loaded = Path(data_path)
datasets = list(data_path_loaded.glob('*'))

st.sidebar.header("Data source")
st.sidebar.success(f"We have {len(datasets)} languages in our datasets.")
st.sidebar.success(f"Data are located at '{data_path}'")

current_lang = st.sidebar.selectbox('Select a language: ', 
                                    [str(x).split('/')[-1] for x in datasets])

col1.header(f"Check all datasets for: {current_lang}")

# Experimentation: display data source

lang_path = Path(f'{data_path}/{current_lang}')
lang_datasets = [str(x).split('/')[-1] for x in list(lang_path.glob('*'))]


from sangkak_estimators import SangkakPosProjetReader

reader_estimator = SangkakPosProjetReader()

for i, tab in enumerate(col1.tabs(lang_datasets)):
    current_file_path = Path(f'{data_path}/{current_lang}/{lang_datasets[i]}')
    with open(current_file_path) as f:
        lines = tab.slider("number of lines ?", 100, 10000, key=i)
        tab.text(f.read(lines)+"\n ...")

    augment = tab.checkbox("Enable augmentation ?", value=False, key=i+10)
    list_train_data, pd_train_data = reader_estimator.fit(current_file_path)\
                                            .transform_analysis(augment=augment)
    tab.dataframe(pd_train_data.describe())

col1.header(f"Check one dataset for: {current_lang}")

current_dataset = col1.selectbox('Select a dataset: ', lang_datasets)

dev_data_path   = lang_path / 'dev.txt'
train_data_path = lang_path / 'train.txt'
test_data_path  = lang_path / 'test.txt'

current_file_path = lang_path / current_dataset


list_train_data, pd_train_data = reader_estimator.fit(current_file_path)\
                                            .transform_analysis(augment=False)


# Experimentation: display models types

models_path = "./experimentations/models"
models_path_loaded = Path(models_path)
models = list(models_path_loaded.glob('*'))

st.sidebar.header("Model type sources")
st.sidebar.success(f"Models are located at '{models_path}'")

current_lang = st.sidebar.selectbox('Select a model: ', 
                                    [str(x).split('/')[-1] for x in models])



# Experimentation: display evaluation models