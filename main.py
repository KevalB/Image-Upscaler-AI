import argparse
from src.train import train_model
from src.predict import upscale_image

def main():
    parser = argparse.ArgumentParser(description='Image Upscaler')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                        help='Mode: train or predict')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        upscale_image()

if __name__ == '__main__':
    main()
