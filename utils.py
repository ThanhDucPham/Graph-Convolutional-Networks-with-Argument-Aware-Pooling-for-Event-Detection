import pickle
import os
import time
import re
import  sys
import numpy as np
import gensim
import json
import torch
def load_bin_vec(fname, vocab):
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
    word_vecs = np.zeros((len(vocab), 300))
    count = 0
    vocab_bin = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(os.path.dirname(__file__), fname), binary=True)
    for word in vocab:
        if word in vocab_bin:
            count += 1
            word_vecs[vocab.index(word)]=(vocab_bin[word])
        else:
            word_vecs[vocab.index(word)] = (np.random.uniform(-0.25, 0.25, 300))
        print("found %d" %count)
    return word_vecs

def load_vocab(filename, hasPad=True):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one sentence_ per line.

    Returns:
        d: dict[sentence_] = index

    """
    d = dict()
    if hasPad:
        d.update({'PAD': 0})
    with open(filename, encoding='utf-8-sig') as f:
        data = f.read().split()
        for id, w in enumerate(data):
            d[w] = len(d)

    return d

def load_trimmed_word2vec(path):
    """
    Load sentence_ embedding Word2vec from file
    :param path: path to the word2vec vectors
    :return:
        vocab: length of vocabulary
        word2id, id2word: dictionary
        word_ebeddings_matrix: contain the vector embedding of each sentence_
    """
    start = time.time()
    print('==> Loading model word2vec...')
    word2id = {}
    id2word = {}
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        word_Embeddings_matrix = [[0]*int(data[0].split(' ')[1])]
        for i, line in enumerate(data[1:len(data)-1]):
            if(line ==''):
                continue
            word_vec = line.split(' ')
            word2id[word_vec[0]]= len(word2id)+1
            word_Embeddings_matrix.append([ float(val) for val in word_vec[1:]])

    id2word = dict(zip(word2id.values(),word2id.keys()))
    nwords = len(word_Embeddings_matrix)
    print('==> Finish load model ({},{})in {:.2f} sec.'.format(nwords, len(word_Embeddings_matrix[0]),time.time()-start))

    return nwords, word2id, id2word, np.array(word_Embeddings_matrix)

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def load_data(window, label):
    vectors = pickle.load(open("vector.bin", 'rb'))
    sents = pickle.load(open(window, 'rb'))
    anchor = pickle.load(open(label, 'rb'))
    return vectors, sents, anchor


def create_document_iter(tokens):
    for doc in tokens:
        raw_doc = ""
        for word in doc:
            raw_doc += " " + word
        yield raw_doc.strip()



def encode_window(tokens, anchors, entities, deps, word2id=None, event2id=None, entity2id=None, window_size=25, save=False, prefix='data/test_'):
    w_windows, ori_words, e_windows, dep_windows, inv_dep_windows, labels, len_sents = [], [], [], [], [], [], []
    unk_id = word2id["UNK"]
    none_e_id = entity2id[NONE]
    pad_id = word2id[PAD]
    epad_id = entity2id[PAD]
    j = 0
    for sent, entities_sent, deps_sent in zip(tokens, entities, deps):

        for tok in np.arange(len(sent)):
            w_window, e_window = [], []
            check_over_top = False
            check_over_bot = False
            for i in range(-window_size, window_size+1):
                if i + tok < 0:
                    w_window.append(pad_id)
                    e_window.append(epad_id)
                    check_over_bot = True

                elif i + tok >= len(sent):
                    check_over_top = True
                    w_window.append(pad_id)
                    e_window.append(epad_id)
                else:
                    w_window.append(word2id.get(sent[i + tok].lower(), unk_id))
                    e_window.append(entity2id.get(entities_sent[i+tok], none_e_id))

            win_deps = []
            win_inv_deps = []
            for pair_dep in deps_sent:
                if all(tok - window_size <= pos <= tok + window_size for pos in pair_dep):
                    win_deps.append([pair_dep[0]-tok+window_size, pair_dep[1]-tok+window_size])
                    win_inv_deps.append([pair_dep[1]-tok+window_size, pair_dep[0]-tok+window_size])
            # print(deps_sent)
            # print('-> ', win_deps)
            # print(len(win_deps))

            len_sents.append([window_size-tok if check_over_bot else 0.,
                              len(sent) - tok + window_size if check_over_top else window_size * 2])
            w_windows.append(w_window)
            e_windows.append(e_window)
            dep_windows.append(win_deps)
            inv_dep_windows.append(win_inv_deps)
            labels.append(event2id[anchors[j][tok]])


        j += 1
    # print(sys.getsizeof(w_windows))
    if save:
        with open(prefix + 'w_window.pkl', 'wb') as f:
            pickle.dump(w_windows, f)
        with open(prefix + 'dep_window.pkl', 'wb') as f:
            pickle.dump(dep_windows, f)
        with open(prefix + 'inv_dep_window.pkl', 'wb') as f:
            pickle.dump(inv_dep_windows, f)
        with open(prefix + 'e_window.pkl', 'wb') as f:
            pickle.dump(e_windows, f)
        with open(prefix + 'label_window.pkl', 'wb') as f:
            pickle.dump(labels, f)

    return w_windows, e_windows, dep_windows, inv_dep_windows, labels, len_sents

def encode_window2(tokens, anchors, entities, deps, word2id=None, event2id=None, entity2id=None, window_size=25, save=False, prefix='data/test_'):
    w_windows, ori_word_windows, e_windows, dep_windows, inv_dep_windows, labels, len_sents = [], [], [], [], [], [], []
    unk_id = word2id["UNK"]
    none_e_id = entity2id[NONE]
    pad_id = word2id[PAD]
    enpad_id = entity2id[PAD]
    evenpad_id = event2id[PAD]

    j = 0
    for sent, entities_sent, deps_sent, label in zip(tokens, entities, deps, anchors):
        w_windows.append([word2id.get(w.lower(), unk_id) for w in sent[:window_size]] + [pad_id] * (window_size - len(sent)))
        ori_word_windows.append(sent[:window_size])
        e_windows.append([entity2id[e] for e in entities_sent[:window_size]] + [enpad_id]* (window_size - len(sent)))
        labels.append([event2id[e] for e in label[:window_size]] + [evenpad_id] * (window_size - len(sent)))

        deps_win, inv_deps_win = [], []
        for pair_dep in deps_sent:
            if pair_dep[0]<window_size and pair_dep[1]<window_size:
                deps_win.append(pair_dep)
                inv_deps_win.append((pair_dep[1], pair_dep[0]))

        dep_windows.append(deps_win)
        inv_dep_windows.append(inv_deps_win)


    # print(sys.getsizeof(w_windows))
    if save:
        with open(prefix + 'w_window.pkl', 'wb') as f:
            pickle.dump(w_windows, f)
        with open(prefix + 'dep_window.pkl', 'wb') as f:
            pickle.dump(dep_windows, f)
        with open(prefix + 'inv_dep_window.pkl', 'wb') as f:
            pickle.dump(inv_dep_windows, f)
        with open(prefix + 'e_window.pkl', 'wb') as f:
            pickle.dump(e_windows, f)
        with open(prefix + 'label_window.pkl', 'wb') as f:
            pickle.dump(labels, f)

    print('word_ids: ', w_windows[:3])
    print('entity_ids: ', e_windows[:3])
    print('label_ids', labels[:3])
    print('dependency_edges: ', dep_windows[0])

    return w_windows, e_windows, dep_windows, inv_dep_windows, labels, len_sents

NONE = 'O'
PAD = 'PAD'
convert_token = dict({'-LRB-': '(', '-RRB-': ')'})
def load_data_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
        words_sents, lab_triggers_sents, entities_sents, dep_sents = [], [], [], []
        equalToken = re.compile('==+')

        for item in data:
            words = item['words']
            golden_entities = [(range(en['head']['start'], en['head']['end']), en['entity-type']) for en in item['golden-entity-mentions']]
            entities = [NONE]* len(words)
            for i in range(len(words)):
                for en in golden_entities:
                    if i in en[0]:
                        e_type = en[1].split(':')[-1]  # get tail of entity-type
                        if i == list(en[0])[0]:
                            e_type = 'B-' + e_type
                        else:
                            e_type = 'I-' + e_type

                        entities[i] = e_type
                        break

            deps = []
            for dep in item['stanford-colcc']:
                dep = dep.split('/')
                if dep[0]!='ROOT':
                    deps.append((int(dep[-1].split('=')[1]), int(dep[-2].split('=')[1])))  # ((governor_id, depend_id),...)

            triggers = [NONE] * len(words)
            for ev in item['golden-event-mentions']:
                range_ = list(range(ev['trigger']['start'], ev['trigger']['end']))
                for idx_ev in range_:
                    event_type = ev['event_type'].split(':')[-1]
                    if idx_ev == range_[0]:
                        event_type = "B-" + event_type
                    else:
                        event_type = "I-" + event_type

                    triggers[idx_ev] = event_type

            for i in range(len(words)):
                if words[i] in ['-LRB-', '-RRB-']:
                    words[i] = convert_token[words[i]]
                elif equalToken.search(words[i]):
                    words[i] = '='

            words_sents.append(words)
            lab_triggers_sents.append(triggers)
            dep_sents.append(deps)
            entities_sents.append(entities)

    # print(words_sents)
    # print(lab_triggers_sents)
    # print(entities_sents)
    # print(dep_sents)
    return words_sents, lab_triggers_sents, entities_sents, dep_sents

def load_data_pickle(fpath, max_sent=31):
    """

    :param fpath:
    :param max_sent:
    :return:
        data: includes 5 tensors:
            - word_ids: data_size, max_sent
            - out_adjacency matrix: sparse tensor matrix: data_size, max_sent, max_sent
            - inverse adjacency matrix : inverse edge in depency graph, sparse tensor matrix:
            - entity_ids: data_size, max_sent
            - labels: data_size, max_sent

    """
    data = []
    for name in ['w_window.pkl', 'dep_window.pkl', 'inv_dep_window.pkl', 'e_window.pkl', 'label_window.pkl']:
        with open(fpath+name, 'rb') as f:
            data.append(pickle.load(f))

    def to_matrix(depends):
        deps = []
        for i, dep_s in enumerate(depends):
            deps.extend([(i,) + dep for dep in dep_s])

        adj_idx = torch.LongTensor(deps)

        matrices = torch.sparse.FloatTensor(adj_idx.t(), torch.FloatTensor([1.] * len(adj_idx)),
                                      torch.Size([len(data[1]), max_sent, max_sent])).to_dense()

        return matrices

    data[0] = torch.LongTensor(data[0])
    data[1] = to_matrix(data[1])
    data[2] = to_matrix(data[2])
    data[3] = torch.LongTensor(data[3])
    data[4] = torch.LongTensor(data[4])

    return data


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[1]
    for element in tag_name.split('-')[2:]:
        tag_type +='-'+element
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = dict(zip(tags.values(), tags.keys()))
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def checkChunk(target, predict, ori_sents, vocab_tag, check_result=False):
    # compare target set with predict set have printing all missing label
    correct_preds, total_correct, total_preds = 0., 0., 0.
    lengths = [len(sent) for sent in target]
    totals = []
    for word_, lab_pred, lab, length in zip(ori_sents, predict, target, lengths):
        lab_chunks = set(get_chunks(lab, vocab_tag))
        lab_pred_chunks = set(get_chunks(lab_pred, vocab_tag))
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)
        if check_result =='pred2ori':
            for ori in lab_pred_chunks:
                if ori not in list(lab_chunks & lab_pred_chunks):
                    # if ori[0] in ['Attack','Meet','Transport','Transfer-Ownership','Start-Position']:
                    check= False
                    for pred in lab_chunks:
                        if pred[1] == ori[1]:
                            print('-->check fault: ', ori,'--',pred)
                            print('   ',lab_chunks)
                            print('   ',lab_pred_chunks)
                            trigger_word = ""
                            for i in range(ori[1], ori[2]):
                                trigger_word += word_[i]+' '
                            print('   trigge word: ',trigger_word)

                            full_sentence = ""
                            for i in range(length):
                                full_sentence +=word_[i]+ ' '
                            print('   full sentence: ', full_sentence)
                            check = True
                            break
                    if not check:
                        print('-->check fault: ', ori, '--')
                        print('   ', lab_chunks)
                        print('   ', lab_pred_chunks)
                        trigger_word = ""
                        for i in range(ori[1], ori[2]):
                            trigger_word += word_[i]+' '
                        print('   Trigger word: ',trigger_word)

                        full_sentence =""
                        for i in range(length):
                            full_sentence += word_[i]+' '
                        print('   full sentence: ', full_sentence)

        elif check_result == 'ori2pred':
            for ori in lab_chunks:
                if ori not in list(lab_chunks & lab_pred_chunks):
                    # if ori[0] in ['Attack','Meet','Transport','Transfer-Ownership','Start-Position']:
                    check= False
                    for pred in lab_pred_chunks:
                        if pred[1] == ori[1]:
                            print('-->check fault: ', ori,'--',pred)
                            print('   ',lab_chunks)
                            print('   ',lab_pred_chunks)
                            trigger_word = ""
                            for i in range(ori[1], ori[2]):
                                trigger_word += word_[i]+' '
                            print('   trigge word: ',trigger_word)

                            full_sentence = ""
                            for i in range(length):
                                full_sentence +=word_[i]+ ' '
                            print('   full sentence: ', full_sentence)
                            check = True
                            break
                    if not check:
                        print('-->check fault: ', ori, '--')
                        print('   ', lab_chunks)
                        print('   ', lab_pred_chunks)
                        trigger_word = ""
                        for i in range(ori[1], ori[2]):
                            trigger_word += word_[i]+' '
                        print('   Trigger word: ',trigger_word)

                        full_sentence =""
                        for i in range(length):
                            full_sentence += word_[i]+' '
                        print('   full sentence: ', full_sentence)

    print(collections.Counter(totals).most_common())

    print('\tresult: {}-{}-{}'.format(total_correct, total_preds, correct_preds))
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return p * 100, r * 100, f1 * 100

# @title evaluate func
def evaluate(config, eval_dataset, model, tokenizer, prefix="", check=None):
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)

    print("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ori_sent_ids = None
    model.eval()
    if check is not None:
        eval_dataloader = notebook.tqdm(eval_dataloader)
    identity_matrix = torch.eye(config.max_sent).unsqueeze(0).to(config.device)
    for batch in eval_dataloader:
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            # input model1
            identity_matrix_batch = identity_matrix.repeat(batch[1].shape[0], 1, 1)
            inputs = {"input_ids": batch[0],
                      "input_adj_out": batch[1],
                      "input_adj_inv": batch[2],
                      "input_self": identity_matrix_batch,
                      "input_ners": batch[3],
                      "labels": batch[4]}

            outputs = model(**inputs)
            logits, tmp_eval_loss = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ori_sent_ids = inputs['input_ids'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ori_sent_ids = np.append(ori_sent_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds_ = np.argmax(preds, axis=2)

    label_map = dict(zip(config.vocab_event.values(), config.vocab_event.keys()))

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    ori_sent_ids_list = [[] for _ in range(out_label_ids.shape[0])]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != config.LAB_PAD_ID:
                out_label_list[i].append(out_label_ids[i, j])
                preds_list[i].append(preds_[i, j])
                ori_sent_ids_list[i].append(ori_sent_ids[i, j])
    ori_sents_list = [[id2word[word] for word in sent if  word!=0] for sent in ori_sent_ids_list]

    prec, recall, f1 = checkChunk(out_label_list, preds_list, ori_sents_list, config.vocab_event, check)
    results = {
        "loss": eval_loss,
        "precision": prec,
        "recall": recall,
        "f1": f1,
    }
    print('\t', results)
    # for key in results.keys():
    #    print("   {} = {:.4f}".format(key, results[key]))

    return results


if __name__ == "__main__":
    nwords, word2id, id2word, _ = load_trimmed_word2vec('data/trimmed_word2vec_new.txt')
    event2id = load_vocab('data/vocab_event.txt', False)
    entity2id = load_vocab('data/vocab_ner_tail.txt')
    word2id.update({'PAD': 0})
    event2id.update({'PAD': -100})
    vocab_event = event2id
    # vocab_event = dict({'O' : 0})
    # for key in event2id:
    #     if key[2:] not in vocab_event and key[2:] != '':
    #         vocab_event.update({key[2:] : len(vocab_event)})
    # print(vocab_event)
    for op in ['dev', 'test', 'train']:
        print('-->opt: ', op)
        words_sents, lab_triggers_sents, entities_sents, dep_sents = load_data_json('data/{}.json'.format(op))
        encode_window2(words_sents, lab_triggers_sents, entities_sents, dep_sents, word2id, vocab_event, entity2id, window_size=31, save=False, prefix='data/loaddata/{}_'.format(op))

