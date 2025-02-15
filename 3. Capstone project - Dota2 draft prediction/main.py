import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import joblib
from tabulate import tabulate
import json
from transformers import pipeline


with open("./heroes.json", "r") as f:
    hero_data = json.load(f)
    hero_id_to_name = {str(hero["id"]): hero["localized_name"] for hero in hero_data["heroes"]}
    
with open("./heroes_skills.json", "r", encoding="utf-8") as f:
    heroes_skills_data = json.load(f)
    hero_tag_to_info = {hero["tag"]: hero for hero in heroes_skills_data}


train_dataset_path = "./dota2Train.csv"
df_train = pd.read_csv(train_dataset_path)


non_hero_columns = ['winner', 'cluster_id', 'game_mode', 'game_type']
num_heroes = df_train.shape[1] - len(non_hero_columns)
hero_ids = list(range(1, num_heroes + 1))
hero_names = [hero_id_to_name.get(str(hero_id), f"Unknown_Hero_{hero_id}") for hero_id in hero_ids]
hero_columns = [f'hero_{i}' for i in range(num_heroes)]

df_train.columns = non_hero_columns + hero_names
df_train = df_train.drop(columns=['cluster_id', 'game_mode', 'game_type'])

def filter_valid_drafts(df):
    team_1_heroes = df.iloc[:, 1:].apply(lambda row: (row == 1).sum(), axis=1)
    team_2_heroes = df.iloc[:, 1:].apply(lambda row: (row == -1).sum(), axis=1)
    return df[(team_1_heroes == 5) & (team_2_heroes == 5)]

df = filter_valid_drafts(df_train)


draft_samples = []
labels = []

for _, row in df.iterrows():
    ally_picks = []
    enemy_picks = []

    for hero in hero_names:
        if row[hero] == 1:
            ally_picks.append(hero)
        elif row[hero] == -1:
            enemy_picks.append(hero)

    for i in range(len(ally_picks)):  # create different draft states
        current_state = {hero: 0 for hero in hero_names}
        for picked_hero in ally_picks[:i]:
            current_state[picked_hero] = 1
        for picked_hero in enemy_picks:
            current_state[picked_hero] = -1
        draft_samples.append(list(current_state.values()))
        labels.append(hero_names.index(ally_picks[i]))  # next hero to pick


X = pd.DataFrame(draft_samples, columns=hero_names)
y = np.array(labels)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# train XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    tree_method="hist"  # Faster training
)
model.fit(X, y)

joblib.dump(model, "xgboost_dota_draft_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")


# loading
model = joblib.load("xgboost_dota_draft_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

text_generator = pipeline("text-generation", model="gpt2")

def generate_hero_explanation(hero_name, ally_picks, enemy_picks):
    """
    Uses NLP to generate a custom explanation for why this hero is good for the draft.
    """
    hero_info = next((hero for hero in heroes_skills_data if hero["name"] == hero_name or hero["tag"] == hero_name.lower().replace(" ", "_")), None)
    
    if not hero_info:
        return f"ℹ️ No details available for {hero_name}."
    
    # getting hero details
    hype = hero_info.get("hype", "No hype description available.")
    abilities = ", ".join([ability["name"] for ability in hero_info.get("abilities", [])])
    role = hero_info.get("attributes", {}).get("Role", "Unknown Role")

    # identify synergy heroes
    ally_names = [hero_id_to_name.get(str(h), h) for h in ally_picks]
    enemy_names = [hero_id_to_name.get(str(h), h) for h in enemy_picks]

    # construct an NLP prompt
    prompt = (
        f"Hero: {hero_name} in Dota 2\n"
        f"Abilities: {abilities}\n"
        f"Role: {role}\n"
        f"Ally Heroes: {', '.join(ally_names)}\n"
        f"Enemy Heroes: {', '.join(enemy_names)}\n"
        f"Why is {hero_name} a good here?"
    )

    # use NLP model to generate explanation
    explanation = text_generator(prompt, max_length=400, num_return_sequences=1, pad_token_id=50256)[0]["generated_text"]

    return f"🌟 **{hero_name}**: {hype}\n🛠 **Abilities**: {abilities}\n🎭 **Role**: {role}\n📝 **Why this pick?**: {explanation}"
  
def recommend_next_heroes(current_picks, enemy_picks, top_n=3):
    """
    Given the current draft state (ally picks) and enemy picks,
    predict the best next heroes considering counter picks.
    """
    if len(current_picks) >= 5:
        return "Draft complete: No more heroes can be picked."

    draft_state = {hero: 0 for hero in hero_names}
    for hero in current_picks:
        if hero in draft_state:
            draft_state[hero] = 1
    for hero in enemy_picks:
        if hero in draft_state:
            draft_state[hero] = -1
    
    draft_array = np.array([list(draft_state.values())])
    hero_probs = model.predict_proba(draft_array)[0]
    sorted_heroes = np.argsort(hero_probs)[::-1]  # sort heroes by probability
    
    recommended_heroes = []
    explanations = []
    
    for recommended_hero in sorted_heroes:
        real_hero = label_encoder.inverse_transform([recommended_hero])[0]
        real_hero_name = hero_id_to_name.get(str(real_hero), f"Unknown_Hero_{real_hero}")
        if real_hero_name not in current_picks and real_hero_name not in enemy_picks:
            recommended_heroes.append(real_hero_name)
            explanations.append(generate_hero_explanation(real_hero_name, current_picks, enemy_picks))
            if len(recommended_heroes) == top_n:
                break
    
    return f"🛡 **Recommended Heroes**: {', '.join(recommended_heroes)}\n\n" + "\n\n".join(explanations)



sample_ally_picks = ["Pudge", "Dazzle", "Beastmaster"]
sample_enemy_picks = ["Invoker", "Juggernaut", "Luna", "Ogre Magi"]
print("🔥 Testing Hero Recommendation with Ally and Enemy Picks...")
print("✅ Ally Picks:", sample_ally_picks)
print("❌ Enemy Picks:", sample_enemy_picks)
print(recommend_next_heroes(sample_ally_picks, sample_enemy_picks))