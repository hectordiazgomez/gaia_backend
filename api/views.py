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

@api_view(['POST'])
def train_model(request):
    file = request.FILES['parallel_corpus']
    params = json.loads(request.data['params'])

    src_lang = params['src_lang']
    tgt_lang = params['tgt_lang']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']
    early_stopping_patience = params['early_stopping_patience']
    validation_split = params['validation_split']
    output_dir = params['output_dir']
    model_name = params['model_name']
    use_gpu = params['use_gpu']
    random_seed = params['random_seed']

    dataset = pd.read_csv(file)

    df_train = dataset[dataset.split=='train'].copy()
    df_dev = dataset[dataset.split=='dev'].copy()
    df_test = dataset[dataset.split=='test'].copy()

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def word_tokenize(text):
        return re.findall('(\w+|[^\w\s])', text)

    def preproc(text):
        clean = mpn.normalize(text)
        clean = replace_nonprint(clean)
        clean = unicodedata.normalize("NFKC", clean)
        return clean

    df_train['src_toks'] = df_train[src_lang].apply(tokenizer.tokenize)
    df_train['tgt_toks'] = df_train[tgt_lang].apply(tokenizer.tokenize)
    df_train['src_words'] = df_train[src_lang].apply(word_tokenize)
    df_train['tgt_words'] = df_train[tgt_lang].apply(word_tokenize)

    model.cuda()
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=learning_rate,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

    model.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_losses = []

        for i in trange(0, len(df_train), batch_size):
            batch_df = df_train[i:i+batch_size]
            src_texts = batch_df[src_lang].tolist()
            tgt_texts = batch_df[tgt_lang].tolist()

            try:
                tokenizer.src_lang = f'{src_lang}_Latn'
                tokenizer.tgt_lang = f'{tgt_lang}_Latn'
                src_inputs = tokenizer(src_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length).to(model.device)
                tgt_inputs = tokenizer(tgt_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_output_length).to(model.device)
                tgt_inputs.input_ids[tgt_inputs.input_ids == tokenizer.pad_token_id] = -100

                loss = model(**src_inputs, labels=tgt_inputs.input_ids).loss
                loss.backward()
                epoch_losses.append(loss.item())

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            except RuntimeError as e:
                print(f"Error: {e}")
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    df_dev['awa_translated'] = df_dev['awa'].apply(lambda x: translate(x, f'{src_lang}_Latn', f'{tgt_lang}_Latn')[0])
    bleu_calc = sacrebleu.BLEU()
    chrf_calc = sacrebleu.CHRF(word_order=2)
    bleu_score = bleu_calc.corpus_score(df_dev['awa_translated'].tolist(), [df_dev['awa'].tolist()]).score
    chrf_score = chrf_calc.corpus_score(df_dev['awa_translated'].tolist(), [df_dev['awa'].tolist()]).score

    results = {
        'model_name': model_name,
        'bleu_score': bleu_score,
        'chrf_score': chrf_score
    }

    return Response(results)
