import tensorflow as tf
from models.custom_model import CustomModel
from src.data_loader import load_data

def train_model():
    # Load and preprocess data
    train_data, train_labels = load_data()
    
    # Initialize custom model
    model = CustomModel()
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    # Train model
    model.fit(train_data, train_labels, epochs=10, batch_size=32)
    
    # Save model
    model.save('custom_model.h5')

if __name__ == '__main__':
    train_model()
