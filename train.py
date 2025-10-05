# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model

def main():
    # Load and preprocess data
    print("Loading Boston Housing dataset...")
    df = load_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Initialize and train the model
    print("Training DecisionTreeRegressor...")
    model = DecisionTreeRegressor(random_state=42)
    trained_model = train_model(model, X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    mse = evaluate_model(trained_model, X_test, y_test)
    
    # Display the result
    print(f"\nDecisionTreeRegressor Performance:")
    print(f"Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
