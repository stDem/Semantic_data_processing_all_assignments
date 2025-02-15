import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import LabelEncoder

# Load hero data
with open("./heroes.json", "r") as f:
    hero_data = json.load(f)
    hero_id_to_name = {str(hero["id"]): hero["localized_name"] for hero in hero_data["heroes"]}

# Load training dataset
train_dataset_path = "./dota2Train.csv"
df_train = pd.read_csv(train_dataset_path)

# Identify hero columns
non_hero_columns = ['winner', 'cluster_id', 'game_mode', 'game_type']
num_heroes = df_train.shape[1] - len(non_hero_columns)
hero_ids = list(range(1, num_heroes + 1))
hero_names = [hero_id_to_name.get(str(hero_id), f"Unknown_Hero_{hero_id}") for hero_id in hero_ids]

# Rename columns with hero names
df_train.columns = non_hero_columns + hero_names
df_train = df_train.drop(columns=['cluster_id', 'game_mode', 'game_type'])

# Ensure valid drafts (5 heroes per team)
def filter_valid_drafts(df):
    team_1_heroes = df.iloc[:, 1:].apply(lambda row: (row == 1).sum(), axis=1)
    team_2_heroes = df.iloc[:, 1:].apply(lambda row: (row == -1).sum(), axis=1)
    return df[(team_1_heroes == 5) & (team_2_heroes == 5)]

df = filter_valid_drafts(df_train)

# Generate training data for hero draft recommendation
draft_samples, labels = [], []

for _, row in df.iterrows():
    ally_picks, enemy_picks = [], []

    for hero in hero_names:
        if row[hero] == 1:
            ally_picks.append(hero)
        elif row[hero] == -1:
            enemy_picks.append(hero)

    for i in range(len(ally_picks)):
        current_state = {hero: 0 for hero in hero_names}
        for picked_hero in ally_picks[:i]:
            current_state[picked_hero] = 1
        for picked_hero in enemy_picks:
            current_state[picked_hero] = -1
        draft_samples.append(list(current_state.values()))
        labels.append(hero_names.index(ally_picks[i]))

# Convert to DataFrame
X = pd.DataFrame(draft_samples, columns=hero_names)
y = np.array(labels)

# Encode hero labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train the XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    tree_method="hist"
)
model.fit(X, y)

# Save trained model
joblib.dump(model, "xgboost_dota_draft_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model training complete and saved!")
