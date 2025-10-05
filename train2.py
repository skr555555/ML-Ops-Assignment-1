# train2.py
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_model, evaluate_model

def main():
    # Load and preprocess data using the SAME functions
    print("Loading Boston Housing dataset...")
    df = load_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Initialize and train the model
    print("Training KernelRidge...")
    model = KernelRidge(alpha=1.0, kernel='linear')
    trained_model = train_model(model, X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    mse = evaluate_model(trained_model, X_test, y_test)
    
    # Display the result
    print(f"\nKernelRidge Performance:")
    print(f"Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
