from pathlib import Path


def test_train_store_csv():
    path = Path('../rossmann-store-sales/train_store.csv')
    assert path.is_file()

def test_train_csv():
    path = Path('../rossmann-store-sales/train.csv')
    assert path.is_file()

def test_test_csv():
    path = Path('../rossmann-store-sales/test.csv')
    assert path.is_file()

def test_store_csv():
    path = Path('../rossmann-store-sales/store.csv')
    assert path.is_file()