import easy_dl

if __name__ == "__main__":
    model = easy_dl.EasyDL("temp.csv", layers=2, neurons=[3, 1], activations=["relu", "sigmoid"])
    model.learn()

    # 2 tane w: w1: 3,4  w2: 1,3 b1: 3,1 b2: 1,1
    #test_results = model.test("test.csv")
    #print(test_results)


