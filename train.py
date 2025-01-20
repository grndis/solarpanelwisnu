from ultralytics import YOLO
import torch
import argparse
import logging
from datetime import datetime
import os

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def check_gpu():
    """Check GPU availability and specs"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        logging.info(f"GPU available: {device}")
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
    else:
        logging.warning("No GPU available, using CPU")

def train_model(args):
    """Train the YOLOv8 model"""
    try:
        # Initialize model
        model = YOLO(args.model_type)
        logging.info(f"Model initialized: {args.model_type}")

        # Training configuration
        training_args = {
            'data': 'dataset/data.yaml',  # path to data.yaml
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch_size,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'patience': args.patience,
            'save': True,
            'amp': True,  # Automatic Mixed Precision
            'workers': args.workers,
            'cache': args.cache,
            'project': 'runs/detect',
            'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }

        # Log training configuration
        logging.info("Training configuration:")
        for key, value in training_args.items():
            logging.info(f"{key}: {value}")

        # Start training
        logging.info("Starting training...")
        results = model.train(**training_args)
        logging.info("Training completed successfully")

        # Validation
        if args.validate:
            logging.info("Starting validation...")
            metrics = model.val()
            logging.info(f"Validation metrics: {metrics}")

        return results

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for crack detection')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='input image size')
    parser.add_argument('--model-type', type=str, default='yolov8n.pt',
                        help='model type to use')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of worker threads')
    parser.add_argument('--patience', type=int, default=20,
                        help='epochs to wait for no improvement before early stopping')
    parser.add_argument('--cache', action='store_true',
                        help='cache images for faster training')
    parser.add_argument('--validate', action='store_true',
                        help='perform validation after training')

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Check GPU
    check_gpu()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train model
    try:
        results = train_model(args)
        logging.info("Process completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

