import streamlit as st
import pandas as pd
import joblib

# Load models
hs_model = joblib.load('hs_model.pkl')
transfer_model = joblib.load('transfer_jc_model.pkl')

st.title('BYU Commitment Predictor')

# -------------------------------
# SELECT PLAYER TYPE
# -------------------------------
player_type = st.selectbox(
    'Select Player Type',
    ['High School', 'Transfer / Junior College']
)

st.write('Enter player information below:')

yes_no = {
    'Yes': 'Y',
    'No': 'N'
}

conf_map = {
    'ACC': 'ACC',
    'Big 12': 'B12',
    'Big Ten': 'B1G',
    'SEC': 'SEC',
    'Group of 6/Independent': 'G6I',
    'FCS School': 'FCS',
    'Junior College': 'JC'
}

pos_map_1 = {
    'CB': 'CB',
    'DB/S': 'DB',
    'DL/DE/DT': 'DL',
    'LB': 'LB',
    'OL/OT/OG/OC': 'OL',
    'QB': 'QB',
    'RB': 'RB',
    'TE': 'TE',
    'WR': 'WR'
}

pos_map_2 = {
    'CB/DB': 'CB',
    'DL/DE/DT': 'DL',
    'LB': 'LB',
    'OL/OT/OG/OC': 'OL',
    'QB': 'QB',
    'RB': 'RB',
    'TE': 'TE',
    'WR': 'WR'
}

# ===============================
# HIGH SCHOOL FORM
# ===============================

if player_type == 'High School':

    top247 = st.selectbox('Top 247 Recruit?', yes_no.keys())
    position = st.selectbox('Position', pos_map_1.keys())
    utah = st.selectbox('From Utah?', yes_no.keys())
    distance = st.number_input('Distance from BYU (miles)', min_value=0.0)
    height = st.number_input('Height (inches)', step=1, format="%d")
    weight = st.number_input('Weight (lbs)', step=1, format="%d")
    score = st.number_input('247 Composite Score', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")

    lds = st.selectbox('LDS?', yes_no.keys())
    alumni = st.selectbox('Alumni Connection?', yes_no.keys())
    poly = st.selectbox('Polynesian?', yes_no.keys())

    input_data = pd.DataFrame({
        '247Top': [top247],
        'Position': [position],
        'Utah': [utah],
        'Distance': [distance],
        'Height': [height],
        'Weight': [weight],
        'Score': [score],
        'LDS': [lds],
        'Alumni': [alumni],
        'Poly': [poly]
    })

    model = hs_model

# ===============================
# TRANSFER / JC FORM
# ===============================
else:

    years = st.number_input('Years of Eligibility Remaining', min_value=0)
    top247 = st.selectbox('Top 247 Transfer', yes_no.keys())
    position = st.selectbox('Position', pos_map_2.keys())
    distance = st.number_input('Distance from BYU (miles)', min_value=0.0)

    conf = st.selectbox('Conference', conf_map.keys())

    height = st.number_input('Height (inches)', min_value=60.0, step=1, format="%d")
    weight = st.number_input('Weight (lbs)', min_value=150.0, step=1, format="%d")
    score = st.number_input('247 Composite Score', min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")

    lds = st.selectbox('LDS?', yes_no.keys())
    alumni = st.selectbox('Alumni Connection?', yes_no.keys())
    prev = st.selectbox('Offered in High School?', yes_no.keys())
    poly = st.selectbox('Polynesian?', yes_no.keys())

    input_data = pd.DataFrame({
        'Years': [years],
        '247Top': [top247],
        'Position': [position],
        'Distance': [distance],
        'Conf': [conf],
        'Height': [height],
        'Weight': [weight],
        'Score': [score],
        'LDS': [lds],
        'Alumni': [alumni],
        'Prev': [prev],
        'Poly': [poly]
    })

    model = transfer_model

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("Predict Commitment Probability"):

    prob = model.predict_proba(input_data)[0][1]

    st.subheader(f"Probability of Commitment: {prob:.3f}")
    st.progress(float(prob))

    if prob > 0.7:
        st.success("High likelihood of committing to BYU")
    elif prob > 0.4:
        st.warning("Moderate likelihood")
    else:
        st.error("Low likelihood")