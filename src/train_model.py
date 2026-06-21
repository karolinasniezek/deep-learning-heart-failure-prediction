import pandas as pd

df = pd.read_csv("../data/heart.csv")

df = pd.get_dummies(
    df,
    columns=[
        "Sex",
        "ST_Slope",
        "ExerciseAngina",
        "RestingECG",
        "ChestPainType"
    ]
)

print(df.head())