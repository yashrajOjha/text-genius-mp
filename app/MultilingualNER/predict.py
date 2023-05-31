import torch
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tag2index = {'O': 0,
 'B-PER': 1,
 'I-PER': 2,
 'B-ORG': 3,
 'I-ORG': 4,
 'B-LOC': 5,
 'I-LOC': 6}

index2tag = {0: 'O',
 1: 'B-PER',
 2: 'I-PER',
 3: 'B-ORG',
 4: 'I-ORG',
 5: 'B-LOC',
 6: 'I-LOC'}

num_labels= 7

xlmr_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/tokenizer.json")

skill_cat = {
    'PER':'Person',
    'LOC':'Location',
    'ORG':'Organization'
}

# print("The maximum length for the inputs is {}".format(xlmr_tokenizer.model_max_length))

xlmr_config = AutoConfig.from_pretrained(
    "xlm-roberta-base",
    num_labels=num_labels,
    id2label=index2tag,
    label2id=tag2index
)
model_hin = (RobertaForTokenClassification
        .from_pretrained("xlm-roberta-mner", config=xlmr_config).to(device))

def predict_nertags(text):
    # print(type(text))
    sample_encoding = xlmr_tokenizer([
        text
 
    ], truncation=True, max_length=512)
    sample_dataset = Dataset.from_dict(sample_encoding)
    sample_dataset = sample_dataset.with_format("torch")

    sample_dataloader = DataLoader(sample_dataset, batch_size=1)
    tokens = []
    labels = []
    for batch in sample_dataloader:
        # predict
        with torch.no_grad():
            output = model_hin(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        predicted_label_id = torch.argmax(output.logits, axis=-1).cpu().numpy()
        # create output
        tokens.append(xlmr_tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))
        labels.append([index2tag[i] for i in predicted_label_id[0]])

    print(pd.DataFrame([tokens[0], labels[0]], index=["Tokens", "Tags"]))

    word_skill= {}
    label_skill = {}
    for i in range(len(sample_encoding['input_ids'])):
        word_ids = sample_encoding.word_ids(batch_index=i)
        for j,l in enumerate(labels[i]):
            if l!='O':
                word_skill.setdefault(word_ids[j],[]).append(xlmr_tokenizer.decode(sample_encoding['input_ids'][i][j]))
                label_skill.setdefault(word_ids[j],[]).append(l)

    skill_nertags = {}
    for key,val in word_skill.items():
        skill_word = "".join(word_skill[key])
        skill_tag = skill_cat[label_skill[key][0].split('-')[1]]
        skill_nertags[key] =[skill_word,skill_tag]
    
    print(skill_nertags)
    word_tags = [val for key,val in skill_nertags.items()]
    skill_nertags = {}
    for ele in word_tags:
        skill_nertags[ele[0]] = ele[1]
    return skill_nertags