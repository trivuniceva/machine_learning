import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import re
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    tokens = text.split()

    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                  "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                  'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",

                  'i', 'ili', 'ali', 'pa', 'ako', 'kada', 'kad', 'onda', 'jer', 'sve',
                  'svi', 'svoj', 'svoje', 'svojih', 'svog', 'svojim', 'svojima', 'svoju',
                  'na', 'u', 'iz', 'iznad', 'ispod', 'između', 'iza', 'ispred', 'pored',
                  'preko', 'pored', 'oko', 'kroz', 'okolo', 'iza', 'ispod', 'iznad',
                  'negde', 'nekuda', 'ovde', 'ovamo', 'onde', 'otuda', 'tamo', 'tuda',
                  'tu', 'tuda', 'tamo', 'ovuda', 'kako', 'gde', 'kad', 'šta', 'ko', 'kome',
                  'koga', 'koji', 'koje', 'kojoj', 'kojima', 'kojim', 'kojih', 'kojeg',
                  'čije', 'čiji', 'čijeg', 'svega', 'sveg', 'svej', 'svejedno', 'svejednoj',
                  'zato', 'zbog', 'budući', 'da', 'li', 'ma', 'evo', 'eto', 'takođe', 'nije',
                  'ne', 'nema', 'imati', 'ima', 'bilo', 'bila', 'bilo', 'bile', 'biti', 'bio',
                  'bih', 'bise', 'bit', 'nemoj', 'dok', 'od', 'do', 'sa', 'sam', 'si', 'sebe',
                  'se', 'te', 'ta', 'tu', 'to', 'taj', 'tome', 'toga', 'toj', 'tom', 'tim',
                  'tih', 'tebi', 'meni', 'mu', 'mu', 'mi', 'moj', 'moja', 'moje', 'mojeg',
                  'mojem', 'mojim', 'moji', 'mojih', 'moju', 'svoj', 'svoja', 'svoje', 'svojeg',
                  'svojem', 'svojim', 'svojih', 'svojima', 'svoju', 'njegov', 'njegova',
                  'njegovo', 'njegovi', 'njegove', 'njegovo', 'njegovom', 'njegovim', 'njegove',
                  'njegovih', 'njegova', 'njegovom', 'njegove', 'njen', 'njena', 'njeno',
                  'njene', 'njenoj', 'njeno', 'njenim', 'njenih', 'njenu', 'njenog', 'njenom',
                  'njenima', 'njihov', 'njihova', 'njihovo', 'njihovi', 'njihove', 'njihovom',
                  'njihovim', 'njihovih', 'njihovoj', 'njihovima', 'njihove', 'možda',
                  'svakako', 'svakome', 'svakoj', 'svakom', 'svakoj', 'svakim', 'svakih',
                  'svakim', 'svakog', 'svakome', 'sada', 'samo', 'sem', 's', 'sa', 'su',
                  'sva', 'svak', 'svaki', 'svakog', 'svakoga', 'svakom', 'svakoj', 'svakome',
                  'svakim', 'svakima', 'svako', 'svakoj', 'svakom', 'svakog', 'svakih',
                  'već', 'većina', 'većini', 'većinu', 'veoma', 'više', 'vrlo', 'još', 'još',
                  'uvek', 'unutra', 'unutrašnji', 'unutrašnja', 'unutrašnje', 'unutrašnji',
                  'unutrašnjoj', 'unutrašnjem', 'unutrašnjeg', 'unutrašnjim', 'unutrašnjih',
                  'unutrašnji', 'unutrašnju', 'nju', 'mu', 'mu', 'ga', 'je', 'jeste',
                  'jesam', 'jesi', 'jest', 'smo', 'ste', 'su', 'bih', 'bi', 'bismo', 'biste',
                  'bi', 'budem', 'budeš', 'bude', 'budemo', 'budete', 'budu', 'ću', 'ćeš',
                  'će', 'ćemo', 'ćete', 'će', 'ćemo', 'ćete', 'hoću', 'hoćeš', 'hoće', 'hoćemo',
                  'hoćete', 'hoće', 'želim', 'želiš', 'želi', 'želimo', 'želite', 'žele',
                  'mogu', 'možeš', 'može', 'možemo', 'možete', 'može', 'mora', 'moraš', 'moramo',
                  'moraju', 'moram', 'moraš', 'mora', 'moramo', 'moraju', 'morao', 'morali',
                  'morala', 'moralo', 'mogao', 'mogli', 'mogle', 'mogla', 'moglo', 'mogući',
                  'moguća', 'moguće', 'mogućih', 'mogućim', 'mogućim', 'mogućoj', 'mogućem',
                  'moguću', 'moj', 'moja', 'moje', 'mojeg', 'mojem', 'mojim', 'moji', 'mojih',
                  'mojima', 'moju', 'možeš', 'može', 'možemo', 'možete', 'može', 'mu', 'mu',
                  'mu', 'mu', 'ni', 'na', 'nad', 'nad', 'nađi', 'ne', 'nego', 'neka', 'nekog',
                  'nekom', 'neki', 'nekog', 'nekoga', 'nekoj', 'nekim', 'nekome', 'nekih', 'neku',
                  'nešto', 'ni', 'nije', 'ništa', 'ništa', 'nisi', 'ništa', 'ništa', 'ništa',
                  'ništa', 'ništa', 'niko', 'nikog', 'nikoga', 'nikome', 'niko', 'niko', 'niko',
                  'nikom', 'niko', 'nikoj', 'nikoga', 'nikoje', 'nikojem', 'nikojim', 'nikoju',
                  'nikojih', 'niko', 'ništa', 'ništa', 'ništa', 'niti', 'niti', 'ništa'


                  }

    filtered_tokens = [word for word in tokens if word not in stop_words]

    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def main():
    data = pd.read_json('train.json')

    data['preprocessed_strofa'] = data['strofa'].apply(preprocess_text)

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(data['preprocessed_strofa'])
    X_test = data['zanr']

    X_train, X_test, y_train, y_test = train_test_split(X_train, X_test, test_size=0.2, random_state=42)

    classifiers = {
        "Logistička regresija": LogisticRegression(),
        "Naivni Bajes": MultinomialNB(),
        "Mašine potpornih vektora": SVC()
    }

    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(f"Izveštaj klasifikacije za {name}:")
        print(classification_report(y_test, y_pred))


