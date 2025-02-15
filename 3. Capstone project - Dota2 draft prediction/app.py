from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
from transformers import pipeline
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}) 
# Load model and label encoder
model = joblib.load("xgboost_dota_draft_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


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
hero_names = [hero_id_to_name.get(str(hero_id), f"Hero_{hero_id}") for hero_id in hero_ids]
hero_columns = [f'hero_{i}' for i in range(num_heroes)]

text_generator = pipeline("text-generation", model="gpt2")




def generate_hero_explanation(hero_name, ally_picks, enemy_picks):
    """
    Uses NLP to generate an explanation for why a hero is recommended.
    """
    hero_info = next((hero for hero in heroes_skills_data if hero["name"] == hero_name or hero["tag"] == hero_name.lower().replace(" ", "_")), None)
    
    if not hero_info:
        return f"No details available for {hero_name}."

    hype = hero_info.get("hype", "No hype description available.")
    abilities = ", ".join([ability["name"] for ability in hero_info.get("abilities", [])])
    role = hero_info.get("attributes", {}).get("Role", "Unknown Role")

    # identify synergy heroes
    ally_names = [hero_id_to_name.get(str(h), h) for h in ally_picks]
    enemy_names = [hero_id_to_name.get(str(h), h) for h in enemy_picks]
    
    prompt = (
        f"Dota 2 Hero: {hero_name}\n"
        f"Abilities: {abilities}\n"
        f"Role: {role}\n"
        # f"Ally Heroes: {', '.join(ally_names)}\n"
        # f"Enemy Heroes: {', '.join(enemy_names)}\n"
        f"Why is {hero_name} a good here in Dota 2 game if your team has {', '.join(ally_names)} and enemy's team has {', '.join(enemy_names)}?"
    )

    explanation = text_generator(prompt, max_length=200, num_return_sequences=1, pad_token_id=50256)[0]["generated_text"]

    return f"🌟 **{hero_name}**: {hype}\n🛠 **Abilities**: {abilities}\n🎭 **Role**: {role}\n📝 **Why this pick?**: {explanation}"
  
@app.route("/heroes", methods=["GET"])
def get_heroes():
    print("Sending hero list:", hero_names)
    return jsonify(hero_names)

  
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📥 Received data:", data)  

        if not data:
            return jsonify({"error": "Invalid JSON request."}), 400

        ally_picks = data.get("ally_picks", [])
        enemy_picks = data.get("enemy_picks", [])

        if not isinstance(ally_picks, list) or not isinstance(enemy_picks, list):
            return jsonify({"error": "Invalid input format. Must be lists."}), 400

        print("✅ Ally Picks:", ally_picks)
        print("❌ Enemy Picks:", enemy_picks)
        
        

        draft_state = {hero: 0 for hero in hero_names}
        
        for hero in ally_picks:
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
          real_hero_name = hero_id_to_name.get(str(real_hero), f"Hero_{real_hero}")
          if real_hero_name not in ally_picks and real_hero_name not in enemy_picks:
              recommended_heroes.append(real_hero_name)
              explanations.append(generate_hero_explanation(real_hero_name, ally_picks, enemy_picks))
              if len(recommended_heroes) == 3:
                  break

        print("🔮 Recommended Heroes:", recommended_heroes)

        return jsonify({
          "recommended_heroes": recommended_heroes,
          "explanations": explanations
        })
    
    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({"error": str(e), "recommended_heroes": [], "explanations": []}), 500


if __name__ == "__main__":
    app.run(debug=True)
