import joblib
import pandas as pd

# Load trained model
package = joblib.load("models/best_model.pkl")

model = package["model"]
features = package["features"]

print("\n===== USED CAR PRICE PREDICTION =====")

# Get user inputs
car_name = input("Enter car name: ")

year = int(input("Enter manufacturing year: "))

km_driven = float(
    input("Enter kilometers driven: ")
)

fuel = input(
    "Enter fuel type (Petrol/Diesel/CNG): "
)

seller_type = input(
    "Enter seller type (Individual/Dealer): "
)

transmission = input(
    "Enter transmission (Manual/Automatic): "
)

owner = input(
    "Enter ownership (First Owner/Second Owner/Third Owner): "
)

# Prepare input data
input_data = {
    "name": car_name,
    "year": year,
    "km_driven": km_driven,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner
}

# Match the training feature structure
input_df = pd.DataFrame(
    [{feature: input_data[feature] for feature in features}]
)

# Predict price
predicted_price = model.predict(input_df)[0]

print("\n================================")
print("Car:", car_name)
print(
    "Predicted Selling Price: ₹{:,.2f}".format(
        predicted_price
    )
)
print("================================")
