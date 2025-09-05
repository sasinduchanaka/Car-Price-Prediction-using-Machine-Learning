
import joblib, pandas as pd
pipe = joblib.load("artifacts/best_model.joblib")
sample = {
  "year": 2016, "km_driven":60000, "mileage":19.5, "engine":1197, "max_power":82,
  "seats":5, "torque_nm":113, "fuel":"Petrol", "seller_type":"Individual",
  "transmission":"Manual", "brand":"Maruti", "car_age":2025-2016
}
df = pd.DataFrame([sample])
print("Predicted:", pipe.predict(df)[0])
