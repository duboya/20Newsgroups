from data_loader import data_loader
from data_model import opt_lgb
from data_preprocess import dict_words

if __name__ == '__main__':
    data_loader()
    x_train, x_test, y_train, y_test = dict_words()
    y_pred = opt_lgb(x_train, x_test, y_train, y_test)
