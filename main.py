from read_data import read_data
from preprocessing import preprocess
from sklearn.preprocessing import LabelEncoder
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report

warnings.filterwarnings("ignore")

def main():
    training, test = read_data()
    processed_training = preprocess(training)

    le_model = LabelEncoder()
    processed_training["label"] = le_model.fit_transform(processed_training["label"])
    print("Processed training (head): \n", processed_training.head())
    print("Training data (head): \n", training.head())

    x_train, x_test, y_train, y_test = train_test_split(
        processed_training["processed"],
        processed_training["label"],
        test_size=0.2,
        random_state=42,
        stratify=processed_training["label"]
    )

    print("Shape of x_train: ", x_train.shape)
    print("Shape of x_test: ", x_test.shape)

    clf = Pipeline([
        ('vectorizer_tri_grams', TfidfVectorizer()),
        ('naive_bayes', (RandomForestClassifier()))
    ])

    clf.fit(x_train, y_train)

    #Getting the prediction
    y_pred = clf.predict(x_test)
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()