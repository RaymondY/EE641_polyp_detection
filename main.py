from utils import load_data
from model import BasicUnet
from train import train_basic_unet
from config import DefaultConfig

config = DefaultConfig()


def main():
    print(f"Using device {config.device}")
    print("Loading data...")
    train_loader = load_data(is_train=True)
    test_loader = load_data(is_train=False)
    print("Data loaded.")
    model = BasicUnet()
    print("Start training...")
    train_basic_unet(model, train_loader, test_loader)
    print("Training finished.")


if __name__ == '__main__':
    main()

