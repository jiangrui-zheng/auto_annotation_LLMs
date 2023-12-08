import random

test_ids = set([])

with open("test.csv", "r") as fin:
    line1 = fin.readline()
    for line in fin:
        tokens = line.strip("\n").split("\t")
        test_ids.add(tokens[0])

train_cate2id = {}
label2index = {}
labelidx = 0

with open("../postprocess/all_examples_0601_hate.csv", "r") as fin:
    fin.readline()
    for line in fin:
        tokens = line.strip("\n").split("\t")
        this_id = tokens[0]
        this_cate = tokens[-1]
        if this_cate not in label2index.keys():
            label2index[this_cate] = labelidx
            labelidx += 1
        if this_id not in test_ids:
            train_cate2id.setdefault(this_cate, [])
            train_cate2id[this_cate].append(this_id)

with open("label2idx.csv", "w") as fout:
    for label, idx in label2index.items():
        fout.write(label + "\t" + str(idx) + "\n")

id2desc = {}

with open("../original/cate2guidelines.csv", "r") as fin:
    for line in fin:
        tokens = line.strip("\n").split("\t")
        id2desc[tokens[1]] = tokens[-1]


# print(id2desc)
def process_trailing(this_sent):
    this_sent = this_sent.replace('"', "'")
    this_sent = this_sent.replace('’', "'")
    this_sent = this_sent.replace('”', "'").replace('“', "'")
    return this_sent


for idx in range(4):
    random.seed(42)
    fout4 = open("train_pr" + str(idx) + ".jsonl", "w")

    with open("train.csv", "r") as fin:
        fin.readline()
        for line in fin:
            tokens = line.strip("\n").split("\t")
            this_id = tokens[0]
            this_sent = process_trailing(tokens[1])
            next_float = random.uniform(0, 1)
            label = tokens[-3]
            labelidx = label2index[tokens[-1]]
            try:
                description = " ".join(process_trailing(id2desc[label]).rstrip().split()[:10])
            except KeyError:
                print(tokens[-3], tokens[-1])
            this_sent = this_sent.rstrip()
            if not (this_sent.endswith("!") or this_sent.endswith(",") or this_sent.endswith(".")):
                this_sent = this_sent.rstrip() + "."
            if idx == 0:
                output_str = "{\"prompt\":\"" + this_sent + " \", \"completion\": \" " + label + "\"}\n"
            elif idx == 1:
                output_str = "{\"prompt\":\"" + this_sent + " \", \"completion\": \" " + label + ": " + description + "\"}\n"
            elif idx == 2:
                output_str = "{\"prompt\":\"Generate the category for the following hate speech: " + this_sent + " :\", \"completion\": \"" + label + "\"}\n"
            else:
                output_str = "{\"prompt\":\"" + this_sent.lstrip().rstrip() + " \", \"completion\": \"" + str(
                    labelidx) + "\"}\n"
            fout4.write(output_str)

    fout4.close()
