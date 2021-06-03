import math
import os
from typing import List

import torch
import numpy as np
import logging
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer
from app.modeling import BertForQuestionAnswering, SemanticBertForQAMRProbeC2QInteractionV3
import collections

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


class SemanticBERT_LM_predictions:
    use_gpu = True
    device = torch.device("cpu")
    bert_model = 'bert-base-uncased'
    max_query_length = 64
    max_seq_length = 128
    max_answer_length = 30

    def __init__(self, model_option='conditional-semanticbert'):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.model_option = model_option
        print('current dir', os.getcwd())

        # Load pre-trained model (weights)
        if model_option == 'conditional-semanticbert':
            output_model_file = "/shared/ccgadmin/demos/QuASE/models/p-QuASE(QAMR)/pytorch_model.bin"
            model_state_dict = torch.load(output_model_file, map_location='cpu')
            self.model = BertForQuestionAnswering.from_pretrained(self.bert_model, state_dict=model_state_dict)
        elif model_option == 'semanticbert':
            output_model_file = '/shared/ccgadmin/demos/QuASE/models/s-QuASE(QAMR)/pytorch_model.bin'
            model_state_dict = torch.load(output_model_file, map_location='cpu')
            self.model = SemanticBertForQAMRProbeC2QInteractionV3.from_pretrained(self.bert_model,
                                                                                  state_dict=model_state_dict)
        elif model_option == 'bert(squad)':
            output_model_file = "/shared/ccgadmin/demos/QuASE/models/p-QuASE(SQuAD)/pytorch_model.bin"
            model_state_dict = torch.load(output_model_file, map_location='cpu')
            self.model = BertForQuestionAnswering.from_pretrained(self.bert_model, state_dict=model_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def convert_example_to_features(self, sentence_text, question_text):

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in sentence_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        query_tokens = self.tokenizer.tokenize(question_text)
        if len(query_tokens) > self.max_query_length:
            query_tokens = query_tokens[0:self.max_query_length]
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if self.model_option == 'semanticbert':

            token_to_orig_map = {}
            question_tokens = []
            sentence_tokens = []
            question_tokens.append("[CLS]")
            for token in query_tokens:
                question_tokens.append(token)
            question_tokens.append("[SEP]")

            sentence_tokens.append("[CLS]")

            for i in range(len(all_doc_tokens)):
                token_to_orig_map[len(sentence_tokens)] = tok_to_orig_index[i]
                sentence_tokens.append(all_doc_tokens[i])
            sentence_tokens.append("[SEP]")

            input_ids_question = self.tokenizer.convert_tokens_to_ids(question_tokens)
            input_ids_sentence = self.tokenizer.convert_tokens_to_ids(sentence_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask_question = [1] * len(input_ids_question)
            input_mask_sentence = [1] * len(input_ids_sentence)

            # Zero-pad up to the query length.
            while len(input_ids_question) < self.max_query_length:
                input_ids_question.append(0)
                input_mask_question.append(0)

            # Zero-pad up to the sequence length.
            while len(input_ids_sentence) < self.max_seq_length:
                input_ids_sentence.append(0)
                input_mask_sentence.append(0)

            assert len(input_ids_question) == self.max_query_length
            assert len(input_mask_question) == self.max_query_length
            assert len(input_ids_sentence) == self.max_seq_length
            assert len(input_mask_sentence) == self.max_seq_length
            return input_ids_question, input_mask_question, input_ids_sentence, input_mask_sentence, doc_tokens, \
                   question_tokens, sentence_tokens, token_to_orig_map
        else:

            tokens = []
            token_to_orig_map = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(len(all_doc_tokens)):
                token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
                tokens.append(all_doc_tokens[i])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            return input_ids, input_mask, segment_ids, doc_tokens, tokens, token_to_orig_map

    def get_predictions(self, start_logits, end_logits, doc_tokens, tokens, token_to_orig_map, n_best_size):

        def _get_best_indexes(logits, n_best_size):
            """Get the n-best logits from a list."""
            index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

            best_indexes = []
            for i in range(len(index_and_score)):
                if i >= n_best_size:
                    break
                best_indexes.append(index_and_score[i][0])
            return best_indexes

        def _compute_softmax(scores):
            """Compute softmax probability over raw logits."""
            if not scores:
                return []

            max_score = None
            for score in scores:
                if max_score is None or score > max_score:
                    max_score = score

            exp_scores = []
            total_sum = 0.0
            for score in scores:
                x = math.exp(score - max_score)
                exp_scores.append(x)
                total_sum += x

            probs = []
            for score in exp_scores:
                probs.append(score / total_sum)
            return probs

        start_indexes = _get_best_indexes(start_logits, n_best_size)
        end_indexes = _get_best_indexes(end_logits, n_best_size)

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["start_index", "end_index", "start_logit", "end_logit"])
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        prelim_predictions = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(tokens):
                    continue
                if end_index >= len(tokens):
                    continue
                if start_index not in token_to_orig_map:
                    continue
                if end_index not in token_to_orig_map:
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > self.max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            tok_tokens = tokens[pred[0]:(pred[1] + 1)]
            orig_doc_start = token_to_orig_map[pred[0]]
            orig_doc_end = token_to_orig_map[pred[1]]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case=True)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        if self.model_option == 'semanticbert':
            probs = np.array(total_scores) / 2
        else:
            probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        return nbest_json

    def calculate_bert_masked_beam_search(self, sent1, sent2, beam_size):
        if self.model_option == 'semanticbert':
            input_ids_question, input_mask_question, input_ids_sentence, input_mask_sentence, doc_tokens, \
            question_tokens, sentence_tokens, token_to_orig_map = self.convert_example_to_features(sent1, sent2)
            input_ids_question = torch.tensor([input_ids_question])
            input_mask_question = torch.tensor([input_mask_question])
            input_ids_sentence = torch.tensor([input_ids_sentence])
            input_mask_sentence = torch.tensor([input_mask_sentence])
            batch_start_logits, batch_end_logits = self.model(input_ids_question, input_ids_sentence,
                                                              input_mask_question, input_mask_sentence)
        else:
            input_ids, input_mask, segment_ids, doc_tokens, tokens, token_to_orig_map = \
                self.convert_example_to_features(sent1, sent2)

            input_ids = torch.tensor([input_ids])
            segment_ids = torch.tensor([segment_ids])
            input_mask = torch.tensor([input_mask])

            batch_start_logits, batch_end_logits = self.model(input_ids, segment_ids, input_mask)
        start_logits = batch_start_logits[0].detach().cpu().tolist()
        end_logits = batch_end_logits[0].detach().cpu().tolist()
        if self.model_option == 'semanticbert':
            nbest_json = self.get_predictions(start_logits, end_logits, doc_tokens, sentence_tokens, token_to_orig_map,
                                              beam_size)
        else:
            nbest_json = self.get_predictions(start_logits, end_logits, doc_tokens, tokens, token_to_orig_map, beam_size)
        print(nbest_json)
        return nbest_json


if __name__ == '__main__':
    BLM = SemanticBERT_LM_predictions()
    output = BLM.calculate_bert_masked_beam_search("abc", "@ and @ is located in USA. ", beam_size=5)
    print(output)
