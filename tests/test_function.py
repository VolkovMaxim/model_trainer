from model_trainer import volkov


def test_volkov():
    assert volkov.fit() == "some result"
