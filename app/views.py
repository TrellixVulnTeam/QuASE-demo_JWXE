import json
import os
import random

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from app.bert_lm import SemanticBERT_LM_predictions

BLM_Conditoinal_SemanticBERT = SemanticBERT_LM_predictions('conditional-semanticbert')
BLM_SemanticBERT = SemanticBERT_LM_predictions('semanticbert')
BLM_BERT_SQuAD = SemanticBERT_LM_predictions('bert(squad)')



@csrf_exempt


def semantic_bert_demo(request):
    context = {
        "alg": "best",
        "model": "conditional-semanticbert"
    }

    return render(request, 'semantic_bert.html', context)


@csrf_exempt
def semantic_bert_calculations(request, sent1, sent2, alg, model):
    if model == 'conditional-semanticbert':
        print('model: Conditional SemanticBERT')
        if alg == "best":
            print("The best prediction")
            predictedTokens = BLM_Conditoinal_SemanticBERT.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
            predictedTokens = predictedTokens[0:1]
        elif alg == "topn":
            print("Top N prediction")
            predictedTokens = BLM_Conditoinal_SemanticBERT.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
            predictedTokens = predictedTokens[0:5]
    elif model == 'semanticbert':
        print('model: SemanticBERT')
        if alg == "best":
            print("The best prediction")
            predictedTokens = BLM_SemanticBERT.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
            predictedTokens = predictedTokens[0:1]
        elif alg == "topn":
            print("Top N prediction")
            predictedTokens = BLM_SemanticBERT.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
            predictedTokens = predictedTokens[0:5]
    elif model == 'bert(squad)':
        print('model: BERT pre-trained on SQuAD')
        if alg == "best":
            print("The best prediction")
            predictedTokens = BLM_BERT_SQuAD.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
            predictedTokens = predictedTokens[0:1]
        elif alg == "topn":
            print("Top N prediction")
            predictedTokens = BLM_BERT_SQuAD.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
            predictedTokens = predictedTokens[0:5]

    table = []
    for rowId in range(len(predictedTokens)):
        row = [(predictedTokens[rowId]['text'], predictedTokens[rowId]['probability'])]
        table.append(row)

    context = {
        "predictedTokens": predictedTokens,
        "table": table,
        "sent1": sent1,
        "sent2": sent2,
        "alg": alg,
        'model': model
    }
    return render(request, 'semantic_bert.html', context)
