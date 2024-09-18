import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

st.set_page_config('Previsão Preço Seguro Médico', page_icon=':material/medical_information:')

st.image('./imagem/logotipo.png')
st.title('Seguro Médico')
st.subheader('Previsão do valor')

def train_model(data):
    X = data.drop(columns='charges')
    y = data['charges']
    
    # Transformação de variáveis categóricas
    X = pd.get_dummies(X, columns=['sex', 'smoker', 'region', 'activity_level', 'pre_existing_conditions'], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    return model, X.columns

# Carregar os dados
data = pd.read_csv('insurance.csv')

# Adicionar as novas colunas para os novos campos
data['activity_level'] = np.random.choice(['sedentário', 'moderado', 'ativo'], size=len(data))
data['pre_existing_conditions'] = np.random.choice(['nenhum', 'diabetes', 'hipertensão'], size=len(data))

# Treinar o modelo e obter as colunas usadas no treinamento
model, columns = train_model(data)

with st.sidebar:
    st.title('Informações')
    
    # Inputs do usuário
    age = st.number_input('Idade', min_value=0, max_value=120, step=1)
    sex = st.selectbox('Sexo', ['male', 'female'])
    bmi = st.number_input('IMC', min_value=0.0, max_value=70.0, step=0.1)
    children = st.number_input('Número de Filhos', min_value=0, max_value=10, step=1)
    smoker = st.selectbox('Fumante', ['yes', 'no'])
    region = st.selectbox('Região', ['southeast', 'southwest', 'northeast', 'northwest'])

    # Novo campo: Nível de atividade física
    activity_level = st.selectbox('Nível de Atividade Física', ['sedentário', 'moderado', 'ativo'])

    # Novo campo: Histórico de doenças pré-existentes
    pre_existing_conditions = st.multiselect('Doenças Pré-existentes', 
                                             ['nenhum', 'diabetes', 'hipertensão'],
                                             default='nenhum')

    # Transformar doenças pré-existentes em variáveis dummies
    condition_diabetes = 1 if 'diabetes' in pre_existing_conditions else 0
    condition_hipertension = 1 if 'hipertensão' in pre_existing_conditions else 0
    condition_nenhum = 1 if 'nenhum' in pre_existing_conditions else 0

    # Prever o valor do seguro
    if st.button('Prever'):
        # Preparar os dados de entrada
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex_male': [1 if sex == 'male' else 0],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0],
            'activity_level_moderado': [1 if activity_level == 'moderado' else 0],
            'activity_level_ativo': [1 if activity_level == 'ativo' else 0],
            'activity_level_sedentário': [1 if activity_level == 'sedentário' else 0],
            'pre_existing_conditions_diabetes': [condition_diabetes],
            'pre_existing_conditions_hipertensão': [condition_hipertension],
            'pre_existing_conditions_nenhum': [condition_nenhum]
        })
        
        # Adicionar colunas que estão faltando no input_data em relação ao conjunto de treinamento
        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Preencher colunas faltantes com 0
                
        # Ordenar as colunas para garantir que estejam na mesma ordem
        input_data = input_data[columns]

        # Fazer a previsão
        prediction = model.predict(input_data) 

        # Exibir o resultado fora da sidebar
        st.session_state['prediction'] = f'O valor previsto do seguro é: ${prediction[0]:,.2f}'

# Verificar e exibir o resultado fora da sidebar
if 'prediction' in st.session_state:
    st.success(st.session_state['prediction'])
