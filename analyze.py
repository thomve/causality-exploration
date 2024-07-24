import matplotlib.pyplot as plt
import seaborn as sns
import torch

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Here is an example sentence to analyze the attention layers of BERT."
input_ids = tokenizer.encode(text, return_tensors='pt')

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

model.eval()

with torch.no_grad():
    outputs = model(input_ids)
    last_layer_attention = outputs.attentions[-1]

print("Shape of last layer attention:", last_layer_attention.shape)

attention_head = last_layer_attention[0][0].detach().numpy()

print("Attention weights for the first head in the last layer:\n", attention_head)

# plt.figure(figsize=(10, 8))
# sns.heatmap(attention_head, cmap='viridis', xticklabels=tokenizer.convert_ids_to_tokens(input_ids[0]), yticklabels=tokenizer.convert_ids_to_tokens(input_ids[0]))
# plt.title('Attention Weights for the First Head in the Last Layer')
# plt.xlabel('Tokens')
# plt.ylabel('Tokens')
# plt.show()