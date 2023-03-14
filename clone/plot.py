from matplotlib import pyplot as plt


def plot_training_stats(train_loss, train_acc):
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label='Modelio mokymo praradimo funkcija')
    plt.plot(range(epochs), train_acc, label='Modelio mokymo tikslumas')
    plt.xlabel('Epocha')
    plt.ylabel('Modelio mokymo parametras')
    plt.title('Modelio mokymo duomenys')
    plt.legend()
    plt.show()

def plot_training_loss_stats(vector):
    epochs = len(vector)
    plt.plot(range(epochs), vector, label='Modelio mokymo praradimo funkcija')
    plt.xlabel('Epocha')
    plt.ylabel('Praradimo funkcija')
    plt.title('Modelio mokymo praradimo funkcijos raida')
    plt.legend()
    plt.show()

def plot_training_acc_stats(vector):
    epochs = len(vector)
    plt.plot(range(epochs), vector, label='Modelio mokymo tikslumo funkcija')
    plt.xlabel('Epocha')
    plt.ylabel('Tikslumas')
    plt.title('Modelio mokymo tikslumas')
    plt.legend()
    plt.show()
