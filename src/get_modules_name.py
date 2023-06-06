from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# write model's named modules into txt file
with open('../flan-t5-base_parameters.txt', 'w') as f:
    for name, _ in model.named_parameters():
        f.write(f'{name}\n')
