import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.forecaster.failure_predictor import FailurePredictor

predictor = FailurePredictor()
predictor.train_forecaster("experiments")

os.makedirs("experiments/forecaster", exist_ok=True)
with open("experiments/forecaster/best_forecaster.pkl", "wb") as f:
    pickle.dump(predictor, f)
print("Forecaster trained and saved to experiments/forecaster/best_forecaster.pkl")
