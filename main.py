from read_data import read_data
from preprocessing import preprocess

def main():
    training, test = read_data()
    processed_training = preprocess(training)
    print(processed_training.head())

if __name__ == "__main__":
    main()