import torch


def main():
    model_filepath = 'output/c/metrics_model.pkl'
    model = torch.load(model_filepath)
    model.eval()
    print('metrics model', model)
    print('metrics model state dict', model.state_dict())
    print('metrics model eval', model.eval())
    print("metrics main finished")

    model_filepath = 'output/c/ast_model.pkl'
    model = torch.load(model_filepath)
    model.eval()
    print('ast model', model)
    print('ast model state dict', model.state_dict())
    print('ast model eval', model.eval())
    print("ast main finished")


if __name__ == "__main__":
    main()
