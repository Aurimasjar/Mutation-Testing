import pandas as pd
from matplotlib import pyplot as plt

"""
Module used to save graphs and their data generated when running genetic algorithm. 
"""

def save_to_csv(obj, filename):
    df = pd.DataFrame(obj)
    df.to_csv('images/data/' + filename + '.csv', index=False)


# object must be the same structure as it was saved with save_to_csv
def read_from_csv(filename):
    return pd.read_csv('images/data/' + filename + '.csv')


def plot_mutation_score_ga(all_scores, non_eq_scores, non_eq_mutants_ratios, filename):
    save_to_csv({'v1': all_scores, 'v2': non_eq_scores, 'v3': non_eq_mutants_ratios}, filename)
    iterations = len(all_scores)
    plt.plot(range(iterations), all_scores, label='visų mutavusių kodų mutavimo įvertis')
    plt.plot(range(iterations), non_eq_scores, label='Neekvivalenčiai pažymėtų mutavusių kodų mutavimo įvertis')
    plt.plot(range(iterations), non_eq_mutants_ratios, label='Neekvivalenčiai pažymėtų mutavusių kodų dalis')
    plt.xlabel('Iteracija')
    plt.ylabel('Mutavusių kodų dalis')
    plt.title('Mutavusių kodų pokytis taikant genetinį algoritmą')
    plt.margins(y=0.45)
    plt.legend(loc='lower left')
    plt.savefig('images/' + filename + '.png')
    plt.show()
