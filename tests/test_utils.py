from src.ml.utils import preprocess_text

def test_preprocess_text_lemmatization():
    result = preprocess_text('Вкусный протеин, мне понравился')

    assert result == 'вкусный протеин понравиться'


def test_preprocess_text_lowercase():
    result = preprocess_text('ОТЛИЧНЫЙ ТОВАР')

    assert result == 'отличный товар'


def test_preprocess_text_punctuation():
    result = preprocess_text('отличный товар!!!')

    assert '!' not in result


def test_preprocess_text_stop_words():
    result = preprocess_text('Плохой и невкусный')

    assert 'и' not in result


def test_preprocess_text_negation():
    result = preprocess_text('не очень')

    assert 'не' in result