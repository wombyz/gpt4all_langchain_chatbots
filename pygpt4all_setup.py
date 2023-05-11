from pygpt4all import GPT4All
model = GPT4All('./models/gpt4all-converted.bin')

for token in model.generate("Once upon a time", n_predict=55):
    print(token, end='', flush=True)
