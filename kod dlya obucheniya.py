from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

DATA_FILE = Path(__file__).with_name("training_phrases.csv")
RANDOM_STATE = 42

NEW_SENTENCES = [
    "Финансовый сервис пересмотрел лимит займа после проверки дохода.",
    "В симуляторе противник меняет тактику в зависимости от поведения игрока.",
    "Система помощи врачу выделяет признаки патологии на снимке КТ.",
    "Учебный портал предлагает повторить задания, в которых учащийся ошибался.",
    "Команда изучает экономику раундов перед отборочным матчем.",
    "Центр мониторинга блокирует подозрительную попытку входа в учетную запись.",
    "Исследователи проверяют состояние популяции животных в национальном парке.",
    "Логист пересчитал время прибытия груза из-за задержки поезда.",
]

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path.name}. Поместите его рядом со скриптом."
        )

    frame = pd.read_csv(path, encoding="utf-8")
    required_columns = {"text", "label"}
    if not required_columns.issubset(frame.columns):
        raise ValueError("В CSV должны быть столбцы 'text' и 'label'.")

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")]

    counts = frame["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Для классификации нужны как минимум два разных класса.")
    if counts.min() < 4:
        raise ValueError(
            "Для каждого класса нужно не менее четырех примеров, чтобы выполнить "
            "стратифицированное разбиение на обучающую и тестовую выборки."
        )
    if frame["text"].duplicated().any():
        duplicates = frame.loc[frame["text"].duplicated(), "text"].tolist()
        raise ValueError(f"Найдены повторяющиеся тексты, удалите дубликаты: {duplicates[:3]}")

    return frame.reset_index(drop=True)

def main() -> None:
    data = load_dataset(DATA_FILE)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(data["label"])
    class_names = encoder.classes_.tolist()

    text_train, text_test, y_train, y_test = train_test_split(
        data["text"].tolist(),
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), max_features=5000, sublinear_tf=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    x_train = vectorizer.fit_transform(text_train)
    x_test = vectorizer.transform(text_test)

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=8,
        learning_rate_init=0.001,
        max_iter=900,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    test_predictions = model.predict(x_test)
    print("Размер набора данных:", len(data), "предложений")
    print("Количество классов:", len(class_names))
    print("Классы:", ", ".join(class_names))
    print("Обучающих примеров:", len(text_train))
    print("Тестовых примеров:", len(text_test))
    print(f"Точность на тестовой выборке: {accuracy_score(y_test, test_predictions):.2%}")
    print("\nОтчет по классам:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("\nПроверка на новых предложениях:")
    new_vectors = vectorizer.transform(NEW_SENTENCES)
    probabilities = model.predict_proba(new_vectors)
    predictions = model.predict(new_vectors)

    for sentence, label_id, row_probabilities in zip(
        NEW_SENTENCES, predictions, probabilities
    ):
        confidence = float(row_probabilities.max())
        print(f"\nПредложение: {sentence}")
        print(f"Предсказанный класс: {class_names[label_id]}")
        print(f"Уверенность модели: {confidence:.2%}")

if __name__ == "__main__":
    main()
