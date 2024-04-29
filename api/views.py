import json
import pandas as pd
import sacrebleu
import torch
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sacremoses import MosesPunctNormalizer
from tqdm.auto import tqdm, trange
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
import re
import sys
import unicodedata
import random
import gc

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

def fix_tokenizer(tokenizer):
    tokenizer.convert_tokens_to_ids = lambda x: tokenizer(x).input_ids[0]
    tokenizer.convert_ids_to_tokens = lambda x: tokenizer.decode([x], skip_special_tokens=True)

def word_tokenize(text):
    return re.findall('(\w+|[^\w\s])', text)

@api_view(['POST'])
def train_model(request):
    file = request.FILES['parallel_corpus']
    params = json.loads(request.POST['params'])

    src_lang = params['src_lang']
    tgt_lang = params['tgt_lang']
    training_steps  = params['training_steps']
    batch_size = params['batch_size']
    src_lang_ws = params["src_lang_ws"]
    tgt_lang_ws = params["tgt_lang_ws"]
    learning_rate = params['learning_rate']
    src_tokenizer = params['src_tokenizer']
    tgt_tokenizer = params["tgt_tokenizer"]
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']
    early_stopping_patience = params['early_stopping_patience']
    writing_system = params["writing_system"]
    model_name = "facebook/nllb-200-distilled-600M"
    random_seed = params['random_seed']

    trans_df = pd.read_csv(file, sep="\t")
    trans_df = trans_df.sample(frac=1, random_state=random_seed)
    df_train = trans_df[:int(len(trans_df)*0.8)]
    df_dev = trans_df[int(len(trans_df)*0.8):]

    tokenizer = NllbTokenizer.from_pretrained(model_name)
    fix_tokenizer(tokenizer)

    smpl = df_train.sample(min(10000, len(df_train)), random_state=random_seed)
    smpl[f'{src_lang}_toks'] = smpl[src_lang].apply(tokenizer.tokenize)
    smpl[f'{tgt_lang}_toks'] = smpl[tgt_lang].apply(tokenizer.tokenize)
    smpl[f'{src_lang}_words'] = smpl[src_lang].apply(word_tokenize)
    smpl[f'{tgt_lang}_words'] = smpl[tgt_lang].apply(word_tokenize)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    model.train()

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=learning_rate,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

    LANGS = [(src_lang, f'{src_lang}_{writing_system}'), (tgt_lang, f'{tgt_lang}_{writing_system}')]

    def get_batch_pairs(batch_size, data=df_train):
        (l1, long1), (l2, long2) = random.sample(LANGS, 2)
        xx, yy = [], []
        for _ in range(batch_size):
            item = data.iloc[random.randint(0, len(data)-1)]
            xx.append(preproc(item[l1]))
            yy.append(preproc(item[l2]))
        return xx, yy, long1, long2
    
    losses = []
    training_steps = training_steps

    x, y, loss = None, None, None
    cleanup()

    best_bleu = 0
    no_improvement = 0
    
    for i in trange(training_steps):
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length).to(model.device)
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_output_length).to(model.device)
            
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

        if i % 1000 == 0:
            print(i, np.mean(losses[-1000:]))

        if i % 1000 == 0 and i > 0:
            model.eval()
            bleu_score = evaluate(model, tokenizer, df_dev, src_lang, tgt_lang)
            model.train()
            
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                no_improvement = 0
                model.save_pretrained(model_name)
                tokenizer.save_pretrained(model_name)
            else:
                no_improvement += 1
                if no_improvement >= early_stopping_patience:
                    print("Early stopping")
                    break
    
    model.eval()
    bleu_score = evaluate(model, tokenizer, df_dev, src_lang, tgt_lang, writing_system )
    chrf_score = evaluate(model, tokenizer, df_dev, src_lang, tgt_lang, writing_system, metric='chrf')

    return Response({
        'model_name': model_name,
        'bleu_score': bleu_score,
        'chrf_score': chrf_score
    })

def evaluate(model, tokenizer, data, src_lang, tgt_lang, writing_system, metric='bleu'):
    translations = []
    references = []

    for _, row in data.iterrows():
        translation = translate(row[src_lang], writing_system, src_lang=f'{src_lang}_{writing_system}', tgt_lang=f'{tgt_lang}_{writing_system}', model=model, tokenizer=tokenizer)
        translations.append(translation[0])
        references.append(row[tgt_lang])

    if metric == 'bleu':
        return sacrebleu.corpus_bleu(translations, [references]).score
    elif metric == 'chrf':
        return sacrebleu.corpus_chrf(translations, [references]).score

def translate(
    text, src_lang='rus_Cyrl', tgt_lang='eng_Latn', 
    a=32, b=3, max_input_length=1024, num_beams=4, model=None, tokenizer=None, **kwargs
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
