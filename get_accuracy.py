import pandas as pd

data = pd.read_csv("/data/jzheng36/auto_annotation/test_with_gpt.csv", sep="\t")

data = data[data['labels']==1]
data['guideline_index'] = data['guideline_index'].astype(str)
data['prediction'] = data['prediction'].astype(str)

data['is_correct'] = data['guideline_index'] == data['prediction']


accuracy = data['is_correct'].mean()
print(f"Accuracy: {accuracy * 100:.2f}%")
