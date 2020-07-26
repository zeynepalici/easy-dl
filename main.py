import easy_dl


def main():
    model = easy_dl.EasyDL("data_banknote_authentication.csv")
    model.learn()

    print(model.predict("test.csv"))


if __name__ == "__main__":
    main()
