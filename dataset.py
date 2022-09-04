from imports import *


line_dict = pd.read_pickle("../graphmake/line_dict.pkl")


class ComplexDataset(Dataset):
    def __init__(self, examples):
        self.examples = [" ".join(example) for example in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return line_dict[self.examples[idx]]


def read_examples(direction, suffix):
    with open(f"../graphmake/cleaned_text/{direction}_{suffix}", "r") as f:
        examples = [line.split() for line in f.readlines()]

    actives = [line for line in examples if line[-1] == "1"]
    decoys = [line for line in examples if line[-1] == "0"]

    return actives, decoys


def get_dataloaders(direction, batch_size):
    actives, decoys = read_examples(direction, "training_normal")
    np.random.shuffle(actives)
    np.random.shuffle(decoys)

    training_actives = actives[:int(len(actives) * 0.8)]
    validation_actives = actives[int(len(actives) * 0.8):]

    training_decoys = decoys[:int(len(decoys) * 0.8)]
    validation_decoys = decoys[int(len(decoys) * 0.8):]
    training_decoys = training_decoys[:len(training_actives)]

    validation_examples = validation_actives + validation_decoys
    training_examples = training_actives + training_decoys

    np.random.shuffle(training_examples)

    actives, decoys = read_examples(direction, "testing")
    testing_examples = actives + decoys

    training_ds = ComplexDataset(training_examples)
    validation_ds = ComplexDataset(validation_examples)
    testing_ds = ComplexDataset(testing_examples)

    training_dl = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)
    testing_dl = DataLoader(testing_ds, batch_size=batch_size, shuffle=True)

    return training_dl, validation_dl, testing_dl

