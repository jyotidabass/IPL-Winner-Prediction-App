import pandas as pd
import joblib
import warnings
import pickle
warnings.filterwarnings("ignore")
X_test = [{'batting_team': 'Chennai Super Kings',
           'bowling_team': 'Royal Challengers Bangalore',
           'city': 'Bangalore',
           'runs_left': 134,
           'balls_left': 76,
           'wickets': 9,
           'total_runs_x': 172,
           'cur_run_rate': 5.181818181818182,
           'req_run_rate': 10.578947368421053}]
input_df = pd.DataFrame(X_test)
print(input_df)
try:
    with open("pipe.pkl", "rb") as f:
        pipeline = pickle.load(f)
        pipeline.predict_proba(input_df)
except Exception as e:
    print(e)
