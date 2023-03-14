def count_accuracy(true_elements, predicted_elements):
    assert len(true_elements) == len(predicted_elements)
    count = 0
    for i in range(0, len(true_elements)):
        if true_elements[i] == predicted_elements[i]:
            count += 1
    return count / len(true_elements)