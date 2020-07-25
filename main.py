import easy_dl


def main():
    model = easy_dl.EasyDL("dog.csv", layers=3, neurons=[4, 6, 1])
    model.learn()

    test_results = model.test()
    print(test_results)

    print(model.predict([0, 0]))


if __name__ == "__main__":
    main()
