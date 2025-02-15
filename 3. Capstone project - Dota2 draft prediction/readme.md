# 🎮 Dota 2 Draft Prediction

Created by:
- Anastasiia Demidova (anastasiia.demidova@study.thws.de);
- Mark Erokhin (mark.erokhin@study.thws.de)


A web application that helps players **predict the best hero picks** in Dota 2 drafts using **machine learning and NLP**. The app recommends heroes based on **ally and enemy picks** and provides **detailed explanations** using AI.

## 📙 Datasets:
1. Games dataset:

  Each row of the dataset is a single game with the following features (in the order in the vector):
  - Team won the game (1 or -1)
  - Cluster ID (related to location)
  - Game mode (eg All Pick)
  - Game type (eg. Ranked)
  - till end: Each element is an indicator for a hero.
  Value of 1 indicates that a player from team '1' played as that hero and '-1' for the other team.

  Hero can be selected by only one player each game. This means that each row has five '1' and five '-1' values.

  We so we removed the following columns:
  - Cluster ID – represents the region of the game.
  - Game mode – indicates the mode (e.g., All Pick, Captains Mode).
  - Game type – ranked or unranked.

2. heroes.json:

  Consists of heroes' names and IDs.

3. heroes_skills.json:

  Consists of different heroes' info like name, id, description, abiliyies, talents, etc.


## 🌟 Features  
✅ **Hero Draft Prediction** – Get recommended heroes based on the current draft.  
✅ **AI-Generated Explanations** – Understand why a hero is a good pick based on current picks.  
✅ **Interactive UI** – Select heroes by clicking and switch between teams easily.  
✅ **Machine Learning Model** – Uses an **XGBoost model** trained on Dota 2 drafts.  
✅ **NLP Analysis** – Uses a **GPT-based model** to explain hero recommendations.  

---

## 🏗️ Tech Stack  
### **Frontend:**  
- **HTML, CSS, JavaScript** – Vanilla frontend for easy interactions.  

### **Backend:**  
- **Flask** – Python backend to serve predictions.  
- **XGBoost** – Trained ML model for hero recommendations.  
- **Transformers (GPT-2)** – NLP model for explanations.   

---

## 🎮 How It Works  
1️⃣ **Select Heroes** – Click on heroes to assign them to a team.  
2️⃣ **Switch Teams** – Easily toggle between selecting for Radiant or Dire.  
3️⃣ **Get Predictions** – Click **"Get Prediction"** to receive recommended heroes.  
4️⃣ **Read Explanations** – See why the AI recommends each hero.  

---

## 🛠️ Installation & Setup  
1️⃣ **Clone repository**  
```bash  
git clone https://github.com/stDem/Dota2_draft_prediction.git  
```
2️⃣ **Install dependencies**  
```bash  
pip install -r requirements.txt  
```
3️⃣ **Run locally**  
```bash  
python app.py
```
4️⃣ **Open index.html**  

open index.html with live server


## Project composition:
- **dota_draft_prediction.ipynb** - Jupyter Notebook includes both models and testing;
- **xgboost_dota_draft_model.pkl, label_encoder.pkl** - saved trained model;
- **dota2Train.csv, heroes.json, heroes_skills.json** - datsets for model training and getting prediction and explnation;
- **main.py, app.py** - for training model and loading it;
- **index.html, script.js, style.css, Space_Grotesk - using for web-application.

---

## 📜 License  
This project is **open-source** under the MIT License.  



## Example:

🔥 Testing Hero Recommendation with Ally and Enemy Picks...<br>
✅ Ally Picks: ['Axe', 'Dazzle']<br>
❌ Enemy Picks: ['Pudge', 'Crystal Maiden']<br><br>
🛡 **Recommended Heroes**: Juggernaut, Enchantress, Earthshaker<br>

🌟 **Juggernaut**: In a flurry of slashes, Juggernaut cuts down his foes. Sprinting and spinning into battle with reckless abandon, and nearly invincible once he is able to begin his assault, stopping Juggernaut can often be just as difficult as surviving him.<br>
🛠 **Abilities**: Blade Fury, Healing Ward, Blade Dance, Omnislash<br>
🎭 **Role**: Carry,Pusher,Escape<br>
📝 **Why this pick?**: Hero: Juggernaut in Dota 2<br>
Abilities: Blade Fury, Healing Ward, Blade Dance, Omnislash<br>
Role: Carry,Pusher,Escape<br>
Why is Juggernaut a good here in Dota 2 game if your team has Axe, Dazzle and enemy's team has Pudge, Crystal Maiden?<br> This guide focuses just on the abilities that Juggernaut has, so you're not surprised when you learn the combos he gets through your fights, but just to learn how to get the most out of your team's abilities, you need to be ready to be in a close fight on the map if there's no one to backstab. <span style="color:blue">All while practicing your passive, dodging damage, avoiding attacks from teammates and getting the most out of your team.</span> If possible, be prepared to focus on all four skills of Juggernaut first and then try to build up some more team utility and damage before you go on in on your poke, as the skills you will be able to use on

🌟 **Enchantress**: Harmful up close and lethal at a distance, Enchantress skewers foes with attacks imbued to become more damaging the further they fly. Whether inflicting powerful slows on her enemies or charming forest creatures to fight her battles, she is never short of tools to win a fight.<br>
🛠 **Abilities**: Untouchable, Enchant, Nature's Attendants, Impetus<br>
🎭 **Role**: Support,Jungler,Pusher,Durable,Disabler<br>
📝 **Why this pick?**: Hero: Enchantress in Dota 2<br>
Abilities: Untouchable, Enchant, Nature's Attendants, Impetus<br>
Role: Support,Jungler,Pusher,Durable,Disabler<br>
Why is Enchantress a good here in Dota 2 game if your team has Axe, Dazzle and enemy's team has Pudge, Crystal Maiden?<br> <span style="color:blue">Probably because even if your mid has Doomfist and team is using the Doomfist to counter the ADC.</span> With 3 enemy's out and no 1 jungle, the difference between a support and support team who have good early game AD can be hundreds of millions.
Abilities: Enchantress, Nature's Attendants, Impetus, Desperado, Enchanted Titan
Why it's good: <span style="color:blue">The most powerful damage of any champion with an ungodly damage and it's the perfect support role for all your ADC heroes.</span><br>

🌟 **Earthshaker**: Whether blocking an enemy's escape, dividing their forces, or shattering the ground beneath gathered foes, Earthshaker is at his best when he strikes without warning. Whatever survives the aftershocks still has a swing from his mighty totem to look forward to.<br>
🛠 **Abilities**: Fissure, Enchant Totem, Aftershock, Echo Slam<br>
🎭 **Role**: Support,Initiator,Disabler,Nuker<br>
📝 **Why this pick?**: Hero: Earthshaker in Dota 2<br>
Abilities: Fissure, Enchant Totem, Aftershock, Echo Slam<br>
Role: Support,Initiator,Disabler,Nuker<br>
Why is Earthshaker a good here in Dota 2 game if your team has Axe, Dazzle and enemy's team has Pudge, Crystal Maiden?<br> A good team can easily snowball your team into a situation where you can get past your opponent. In other words, if they are having problems, they have to learn how to play their game and go to face value, not simply get into a bad situation.
In my opinion, <span style="color:blue">Earthshaker is a very good lane pick. Its hard to be successful at this solo lane. To find a solid way to punish opponents that might not be well-positioned or to get the best out of your supports in teamfights, you need to always play your lane. </span>A lot of times a lane swap will mean going on offense

