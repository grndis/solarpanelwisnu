from ultralytics import YOLO
import torch
import argparse

def train_model(epochs=100, batch_size=16, imgsz=640):
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Train model
    results = model.train(
        data='dataset/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=20,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    
    # Validate
    metrics = model.val()
    
    return model, results, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    
    train_model(args.epochs, args.batch, args.imgsz)
