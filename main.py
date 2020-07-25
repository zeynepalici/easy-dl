import easy_dl


def main():
    model = easy_dl.EasyDL("dog.csv", layers=3, neurons=[4, 6, 1], iterations=5000)
    model.learn()

    # test_results = model.test()
    # print(np.round(test_results))

    print(model.predict([8, 7]))
        print("sictik")


if __name__ == "__main__":
    main()
