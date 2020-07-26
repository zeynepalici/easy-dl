import easy_dl


def main():
    model = easy_dl.EasyDL("data_banknote_authentication.csv", layers=3, neurons=[4, 6, 1], iterations=5000)
    model.learn()

    print(model.predict("test.csv"))


if __name__ == "__main__":
    main()
