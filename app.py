import streamlit as st
import pandas as pd
import pickle
import sklearn
# Hide dev element
hide_streamlit_style = """
<style>
#MainMenu {
  visibility: hidden;
}
footer {
    visibility: hidden
}

div[data-testid$='stToolbar']{
    visibility: hidden
}
button[data-testid$='manage-app-button']{
    visibility: hidden
}
<style/>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

# declaring the venues

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']


pipe = pickle.load(open('model.pkl', 'rb'))
st.title('IPL Win Predictor')


col1, col2 = st.columns(2)

with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))

with col2:

    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))


city = st.selectbox(
    'Select the city where the match is being played', sorted(cities))


target = st.number_input('Target', step=1, min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', step=1, min_value=0)

with col4:
    overs = st.number_input('Overs Completed', step=1, min_value=0)

with col5:
    wickets = st.number_input('Wickets Fallen', step=1, min_value=0)


if st.button('Predict Probability'):
    try:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        currentrunrate = score / overs
        requiredrunrate = (runs_left * 6) / balls_left

        if runs_left <= 0:
            st.header(battingteam + "- " + str(round(1 * 100)) + "%")
            st.header(bowlingteam + "- " + str(round(0 * 100)) + "%")
        else:
            input_df = pd.DataFrame({'batting_team': [battingteam], 'bowling_team': [bowlingteam], 'city': [city], 'runs_left': [runs_left], 'balls_left': [
                                    balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'cur_run_rate': [currentrunrate], 'req_run_rate': [requiredrunrate]})

            result = pipe.predict_proba(input_df)
            lossprob = result[0][0]
            winprob = result[0][1]
            st.header(battingteam + "- " + str(round(winprob * 100)) + "%")
            st.header(bowlingteam + "- " + str(round(lossprob * 100)) + "%")

    except Exception as e:
        st.warning('Error, please check if all inputs are valid', icon="⚠️")
