#Используя модель RuGPT от Сбера, необходимо реализовать возможность генерации текста по заданному промпту.
# Допускается использование как старой модели rugpt3large_based_on_gpt2, так и новой ruGPT-3.5-13B.
# Каждому студенту преподавателем будет дана пара слов, и требуется подобрать промпт таким образом,
# чтобы выданная пара слов встречалась бы в сгенерированном тексте с учётом порядка и с учётом словоформ (как в предыдущей работе).
# Допускается использовать обе модели, но пара слов преподавателем подбирается на rugpt3large_based_on_gpt2.
# Ограничений на значения параметров нет (даже на длину генерируемого текста), но подобранный промпт не должен содержать искомые слова ни в одной из их словоформ.
# Горы, тайга
from sys import argv
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
np.random.seed(42)
torch.manual_seed(42)

def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path ), GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Альтернативно "ai-forever/ruGPT-3.5-13B"

tok, model = load_tokenizer_and_model ("sberbank-ai/rugpt3large_based_on_gpt2")

def generate(
            model, tok, text,
            do_sample=True, max_length=500, repetition_penalty=5.0,
            top_k=5, top_p=0.95, temperature=1,
            num_beams=None,
            no_repeat_ngram_size=3
            ):
          input_ids = tok.encode(text, return_tensors="pt")
          print(model.generate.__globals__['__file__'])
          out = model.generate(
              input_ids,
              max_length=max_length,
              repetition_penalty=repetition_penalty,
              do_sample=do_sample,
              top_k=top_k, top_p=top_p, temperature=temperature,
              num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
              )
          return list(map(tok.decode, out))

#text = "Одинокий путник присел на холме и принялся дотошливо зарисовывать сибирский пейзаж, раскинувшийся перед ним - "
text = "Прогулки по Алтаю "
generated = generate(model, tok, text = text, num_beams=10)

print(generated[0])
