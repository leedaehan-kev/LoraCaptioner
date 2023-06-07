from transformers import AutoModelForSeq2SeqLM


def write_named_parameters(checkpoint, output_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # write model's named modules into txt file
    with open(output_path, 'w') as f:
        for name, _ in model.named_parameters():
            f.write(f'{name}\n')
