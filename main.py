import easy_dl
import matplotlib.pyplot as plt


def main():
    model = easy_dl.EasyDL("data_banknote_authentication.csv")
    cost_history = model.learn()

    plt.plot(cost_history)
    plt.show()

    print(f"Train Accuracy: {model.evaluate()[1]}")
    # print(model.predict("test.csv"))


if __name__ == "__main__":
    main()
