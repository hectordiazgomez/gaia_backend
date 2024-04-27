import pandas as pd
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm, trange
import numpy as np
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
import re
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
import random
import gc
import torch

#output.csv comes from the frontend
dataset = pd.read_csv('output.csv')
print(dataset.head())

df_train = dataset[dataset.split=='train'].copy()
df_dev = dataset[dataset.split=='dev'].copy()
df_test = dataset[dataset.split=='test'].copy()

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#The language code are the first 3 letter of the language name  + _Latn
tokenizer.src_lang = "spa_Latn"
inputs = tokenizer(text="Hola a todos mis amigos", return_tensors="pt")
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
)
print(tokenizer.decode(translated_tokens[0], skip_special_tokens=True))

def word_tokenize(text):

    return re.findall('(\w+|[^\w\s])', text)
#The language token codes are the first 3 letter of the language name  + _toks
#The language words code are the first 3 letter of the language name  + _words
smpl = df_train.sample(5, random_state=1)
smpl['spa_toks'] = smpl.spa.apply(tokenizer.tokenize)
smpl['awa_toks'] = smpl.awa.apply(tokenizer.tokenize)
smpl['spa_words'] = smpl.spa.apply(word_tokenize)
smpl['awa_words'] = smpl.awa.apply(word_tokenize)

stats = smpl[
    ['spa_toks', 'awa_toks', 'spa_words', 'awa_words']
].applymap(len).describe()
print(stats.spa_toks['mean'] / stats.spa_words['mean'])
print(stats.awa_toks['mean'] / stats.awa_words['mean'])
stats

texts_with_unk = [
    text for text in tqdm(dataset.awa)
    if tokenizer.unk_token_id in tokenizer(text).input_ids
]
print(len(texts_with_unk))
s = random.sample(texts_with_unk, 5)
print(s)

mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]

def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

texts_with_unk_normed = [
    text for text in tqdm(texts_with_unk)
    if tokenizer.unk_token_id in tokenizer(preproc(text)).input_ids
]
print(len(texts_with_unk_normed))

model.cuda();
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

# The language code will be the first 3 letters of the word + _Latn
LANGS = [('spa', 'spa_Latn'), ('awa', 'awa_Latn')]

def get_batch_pairs(batch_size, data=df_train):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2

print(get_batch_pairs(1))

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

batch_size = 16
max_length = 128
training_steps = 300
losses = []
#Save the model under the folder /content/nllb- source language code - target language code - v1
MODEL_SAVE_PATH = '/content/nllb-spa-awa-v2'

model.train()
x, y, loss = None, None, None
cleanup()

tq = trange(len(losses), training_steps)
for i in tq:
    xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
    try:
        tokenizer.src_lang = lang1
        x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
        tokenizer.src_lang = lang2
        y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

        loss = model(**x, labels=y.input_ids).loss
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    except RuntimeError as e:
        optimizer.zero_grad(set_to_none=True)
        x, y, loss = None, None, None
        cleanup()
        print('error', max(len(s) for s in xx + yy), e)
        continue

    if i % 100 == 0:
        print(i, np.mean(losses[-100:]))

    if i % 100 == 0 and i > 0:
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)

model_load_name = '/content/nllb-spa-awa-v2'
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)

#Language codes are the first 3 letters + _Latn
def translate(
    text, src_lang='spa_Latn', tgt_lang='awa_Latn',
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True,
        max_length=max_input_length
    )
    model.eval()
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)

dataset = pd.read_csv('output.csv')
df_dev = dataset[dataset['split'] == 'dev'][:10]

#Code is the first three letters of the target_language_translated
df_dev['awa_translated'] = df_dev['awa'].apply(lambda x: translate(x, 'spa_Latn', 'awa_Latn')[0])
sampled_df = df_dev.sample(10, random_state=5)[["spa", "awa",  "awa_translated"]]

bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)

print(bleu_calc.corpus_score(df_dev['awa_translated'].tolist(), [df_dev['awa'].tolist()]))
print(chrf_calc.corpus_score(df_dev['awa_translated'].tolist(), [df_dev['awa'].tolist()]))
