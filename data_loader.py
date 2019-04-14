## Download the dataset
import tarfile
import urllib.request


def data_loader():
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz", "a.tar.gz")

    # Extracting the dataset
    tar = tarfile.open("a.tar.gz")
    tar.extractall()
    tar.close()
