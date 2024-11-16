from read_data import read_data
from preprocessing import preprocess
from sklearn.preprocessing import LabelEncoder
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report
import pandas as pd

warnings.filterwarnings("ignore")

def main():
    training, test = read_data()
    processed_training = preprocess(training)

    le_model = LabelEncoder()
    processed_training["label"] = le_model.fit_transform(processed_training["label"])
    print("Processed training (head): \n", processed_training.head())

    x_train, x_test, y_train, y_test = train_test_split(
        processed_training["processed"],
        processed_training["label"],
        test_size=0.2,
        random_state=42,
        stratify=processed_training["label"]
    )

    print("\nShape of x_train: ", x_train.shape)
    print("Shape of x_test: ", x_test.shape)

    clf = Pipeline([
        ('vectorizer_tri_grams', TfidfVectorizer()),
        ('naive_bayes', (MultinomialNB()))
    ])

    clf.fit(x_train, y_train)

    #Getting the prediction
    y_pred = clf.predict(x_test)
    print("\nAccuracy score: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    classes = training["label"].unique()
    print("We can predict which class a tweet belongs to. We have the following classes: ", classes)

    processed_testing = preprocess(test)
    print("Processed testing (head): \n", processed_testing.head())

    results_csv = pd.DataFrame({
        "Tweet": [],
        "True label": [],
        "Predicted label": []
    })

    for i in range(processed_testing.shape[0]):
        prediction = clf.predict([processed_testing["processed"][i]])

        results_csv = pd.concat([results_csv, pd.DataFrame({
            "Tweet": [processed_testing["text"][i]],
            "True label": [processed_testing["label"][i]],
            "Predicted label": [classes[prediction-1][0]]
        })], ignore_index=True)

    results_csv.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()