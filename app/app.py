from flask import Flask, render_template, request
from recommendations import PROMPT_RECOMMENDATIONS
import random

app = Flask(__name__)

AVAILABLE_MODELS = [
# TODO nazwy modeli
]

def generate_response(prompt, model_name):
    # TODO zmiana na response
    return f"Odpowiedź modelu ({model_name}) na prompt: {prompt}"

def estimate_energy_cost(prompt, response):
    # TODO zmiana na liczenie wedlug wzoru + dodac SHAP
    base_cost = 0.0001 * len(response)
    return round(base_cost, 4)

def get_top_recommendations(prompt):

    # TODO pobrac top 3 wedlug SHAP
    mock_features = random.sample(list(PROMPT_RECOMMENDATIONS.keys()), 3)

    return [PROMPT_RECOMMENDATIONS[f] for f in mock_features]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form.get("prompt_input")
        model_name = request.form.get("model_selector")


        response = generate_response(prompt, model_name)

        energy_cost = estimate_energy_cost(prompt, response)

        recommendations = get_top_recommendations(prompt)

        return render_template(
            "index.html",
            models=AVAILABLE_MODELS,
            response=response,
            energy_cost=energy_cost,
            recommendations=recommendations
        )

    return render_template("index.html", models=AVAILABLE_MODELS)


if __name__ == "__main__":
    app.run(debug=True)
