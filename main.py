import easy_dl

if __name__ == "__main__":
    model = easy_dl.EasyDL("temp.csv", layers=2, units=[3, 3], activations=["relu", "sigmoid"])
    #model.learn()

    #test_results = model.test("test.csv")
    #print(test_results)


