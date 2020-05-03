from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from tqdm import tqdm_notebook, tqdm
import math
import os
import random
import collections
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from Event.GCN_2018.utils import load_vocab

class Config(object):
    def __init__(self):
        self.num_epoch = 5
        self.learning_rate = 0.01
        self.weight_decay = 1e-4
        self.adam_eps = 1e-8
        self.batch_size = 128
        self.eval_batch_size = 128
        self.nstep_logging = 500
        self.warmup_steps= 2000
        self.max_restart = 4
        self.window_size = 15
        self.seed = 150
        self.max_sent = self.window_size * 2 + 1
        self.vocab_word_size = 14078
        self.fine_tune=True
        self.EPAD_ID = 0
        self.WPAD_ID = 0
        self.LAB_PAD_ID = -100
        self.EPAD = 'PAD'
        self.WPAD = 'PAD'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dir_train = 'data/train.json'
        self.dir_dev = 'data/dev.json'
        self.test_dir = 'data/test.json'
        self.dir_word2vec = 'data/trimmed_word2vec_new.txt'
        self.dir_data = 'data/'
        self.output_dir = 'results/gcn_2018/'
        self.load_data()
        try:
            print('Currently working on ', torch.cuda.get_device_name(0))
        except:
            pass


    def load_data(self):
        vocab_event = load_vocab(self.dir_data + 'vocab_event_2.txt', hasPad=False)
        self.vocab_event = dict({'O': 0})
        for key in vocab_event:
            if key[2:] not in self.vocab_event and key[2:] != '':
                self.vocab_event.update({key[2:]: len(self.vocab_event)})
        self.vocab_ner = load_vocab(self.dir_data + 'vocab_ne_2.txt')
        self.num_class_events = len(self.vocab_event)
        self.num_class_entities = len(self.vocab_ner)


    def set_seed(self, seed=None):
        if seed is None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class AttentionLayer(nn.Module):
    def __init__(self, D, H=128, return_sequences=False):
        '''
        A single convolutional unit
        :param D: int, input feature dim
        :param H: int, hidden feature dim
        :param return_sequences: boolean, whether return sequence
        '''
        super(AttentionLayer, self).__init__()

        # Config copying
        self.H = H
        self.return_sequences = return_sequences
        self.D = D
        self.W1 = nn.Linear(D, H)
        self.W2 = nn.Linear(D, H)
        self.V = nn.Linear(H, 1)


    def softmax_mask(self, x, mask):
        '''
        Softmax with mask
        :param x: torch.FloatTensor, logits, [batch_size, seq_len, seq_len, 1]
        :param mask: torch.ByteTensor, masks for sentences, [batch_size, seq_len]
        :return: torch.FloatTensor, probabilities, [batch_size, seq_len, seq_len, 1]
        '''
        x_exp = torch.exp(x)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            x_exp = x_exp * mask.float()
        x_sum = torch.sum(x_exp, dim=-1, keepdim=True) + 1e-16
        x_exp /= x_sum  # batch, seq1, seq2,1
        return x_exp

    def forward(self, x_text, mask, x_attention=None):
        '''
        Forward this module
        :param x_text: torch.FloatTensor, input features, [batch_size, seq_len, D]
        :param mask: torch.ByteTensor, masks for features, [batch_size, seq_len]
        :param x_attention: torch.FloatTensor, input features No. 2 to attent with x_text, [batch_size, seq_len, D]
        :return: torch.FloatTensor, output features, if return sequences, output shape is [batch, SEQ_LEN, D];
                    otherwise output shape is [batch, D]
        '''
        if x_attention is None:
            x_attention = x_text

        x_text = x_text.unsqueeze(2)  # batch, seq, 1, dim ~ query
        x_attention = x_attention.unsqueeze(1) # batch, 1 ,seq, dim  ~ key, value

        scores = self.V(torch.tanh(self.W1(x_attention) + self.W2(x_test)))  # batch, seq, seq, 1
        scores_masked = self.softmax_mask(scores, mask)
        output = (x_attention * scores_masked).sum(-2)

        if not self.return_sequences:
            output = torch.sum(output, -2)
        return output


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.linear = nn.Linear(in_features, out_features)
        self.weight = nn.parameter.Parameter(torch.FloatTensor(out_features, in_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if self.bias is not None:
            support = F.linear(input, self.weight, self.bias)
        else:
            support = F.linear(input, self.weight)
        output = torch.bmm(adj, support)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class EdgeWiseGateLayer(nn.Module):
    def __init__(self, in_features=300, opt='origin'):
        super(EdgeWiseGateLayer, self).__init__()
        self.W = nn.Linear(in_features=in_features, out_features=1)
        self.act = nn.Sigmoid()
        self.opt = opt
        self.in_features = in_features

    def forward(self, h_v):
        s_k = self.act(self.W(h_v))
        return s_k

    def __repr__(self):
        return self.__class__.__name__+'(opt={}, in_feat={} -> out_feat=1'.format(self.opt, self.in_features)


class GCNLayer2(nn.Module):
    def __init__(self, in_features=300, out_features=300):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_out = nn.Linear(in_features=in_features, out_features=out_features)
        self.W_inverse = nn.Linear(in_features, out_features=out_features)
        self.W_self = nn.Linear(in_features, out_features)
        self.gate_out = EdgeWiseGateLayer(in_features, 'origin')
        self.gate_inverse = EdgeWiseGateLayer(in_features, 'inverse')
        self.gate_self = EdgeWiseGateLayer(in_features, 'self')
        self.act = nn.ReLU()

    def forward(self, input, adj_out, adj_inv, adj_self):
        assert adj_out.shape == adj_inv.shape == adj_self.shape, self.print_err(adj_out, adj_self)

        h_out = torch.bmm(adj_out, self.gate_out(input) * self.W_out(input))
        h_inverse = torch.bmm(adj_inv, self.gate_inverse(input) * self.W_inverse(input))
        h_self = torch.bmm(adj_self, self.gate_self(input) * self.W_self(input))

        output = self.act(h_inverse + h_out + h_self)
        return output

    def print_err(self, adj_out, adj_self):

        string = 'get adj_out shape= {}, adj_self shape = {}'.format(adj_out.shape, adj_self.shape)
        return "shape of three adj matrices corresponding with 3 types of edge: origin, inverse, self-attention have to be equal\n"+ string


class PoolLayer(nn.Module):
    def __init__(self, config):
        self.config = config
        super().__init__()

    def forward(self, input_hid, idx_current_word, entity_input):
        if idx_current_word is not None:
            current_word = input_hid[:, idx_current_word].unsqueeze(1)
        else:
            current_word = input_hid[:, config.window_size].unsqueeze(1)

        active_entities = (entity_input != self.config.EPAD_ID).float()
        notNone_entities = (entity_input != 1).float()
        # print(active_entities.shape)
        # print(input_hid.shape)
        entity_vecs = input_hid * active_entities.unsqueeze(-1) * notNone_entities.unsqueeze(-1)
        concate_vec = torch.cat((current_word, entity_vecs), dim=-2)  # batch_size, 1+num_entities, dim_hid
        max_pooling_vec = torch.max(concate_vec, dim=1).values  # batch_size, dim_hid

        return max_pooling_vec


class EDModel(nn.Module):
    def __init__(self, config, pretrained_embeddings=None):
        self.config = config
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_word_size, 300, padding_idx=0, _weight=pretrained_embeddings)
        self.ner_embeddings = nn.Embedding(config.num_class_entities, 50, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_sent, 50)
        self.biLSTM = nn.LSTM(input_size=400, hidden_size=300, num_layers=2, dropout=0.5,
                              batch_first=True, bidirectional=True)

        self.gcn = GCNLayer2(in_features=300*2, out_features=300)
        self.gcn2 = GCNLayer2(in_features=300, out_features=300)

        self.pooler = PoolLayer(config)
        self.dropout = nn.ModuleList()
        for _ in range(4):
            self.dropout.append(nn.Dropout(0.5))

        self.classifier = nn.Linear(in_features=300, out_features=config.num_class_events)


    def forward(self,
                input_ids,
                input_ners,
                input_adj_out,
                input_adj_inv,
                input_self,
                labels=None
                ):
        word_embeddings = self.word_embeddings(input_ids)
        ner_embeddings = self.ner_embeddings(input_ners)

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.config.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = torch.cat((word_embeddings, ner_embeddings, position_embeddings), dim=-1)
        embeddings = self.dropout[0](embeddings)

        bilstm = self.biLSTM(embeddings)
        bilstm = self.dropout[1](bilstm[0])
        gcn_out = self.gcn(bilstm,
                           input_adj_out,
                           input_adj_inv,
                           input_self,)
        gcn_out = self.dropout[2](gcn_out)
        gcn_out = self.gcn2(gcn_out,
                            input_adj_out,
                            input_adj_inv,
                            input_self,)

        pool_out = self.pooler(gcn_out, input_ners)  # (batch_size, hid_dim)
        pool_out = self.dropout[3](pool_out)
        logits = self.classifier(pool_out)

        outputs = (logits,)
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits[active_loss]
            activel_labels = labels[active_loss]
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(active_logits, activel_labels)
            outputs += (loss,)

        return outputs   # (logits, loss)


class EDModel2(nn.Module):
    """
    implement the model from paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16329
    """
    def __init__(self, config, pretrained_embeddings=None):
        self.config = config
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_word_size, 300, padding_idx=0, max_norm=3)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(pretrained_embeddings)
            self.word_embeddings.weight.requires_grad = config.fine_tune
        self.ner_embeddings = nn.Embedding(config.num_class_entities, 50, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_sent, 50)
        self.biLSTM = nn.LSTM(input_size=400, hidden_size=300, num_layers=2, dropout=0.5,
                              batch_first=True, bidirectional=True)

        self.gcn = GCNLayer2(in_features=300*2, out_features=300)
        self.gcn2 = GCNLayer2(in_features=300, out_features=300)

        self.pooler = PoolLayer(config)
        self.dropout = nn.ModuleList()
        for _ in range(4):
            self.dropout.append(nn.Dropout(0.5))

        self.classifier = nn.Linear(in_features=300, out_features=config.num_class_events)


    def get_sentence_positional_feature(self, BATCH_SIZE, SEQ_LEN):
        positions = [[abs(j) for j in range(-i, SEQ_LEN - i)] for i in range(SEQ_LEN)]  # list [SEQ_LEN, SEQ_LEN]
        positions = [torch.LongTensor(position) for position in positions]  # list of tensors [SEQ_LEN]
        positions = [torch.cat([position] * BATCH_SIZE).resize_(BATCH_SIZE, position.size(0))
                     for position in positions]  # list of tensors [BATCH_SIZE, SEQ_LEN]
        return positions


    def forward(self,
                input_ids,
                input_ners,
                input_adj_out,
                input_adj_inv,
                input_self,
                labels=None
                ):
        word_embeddings = self.word_embeddings(input_ids)
        word_embeddings = self.dropout[0](word_embeddings)
        ner_embeddings = self.ner_embeddings(input_ners)
        ner_embeddings = self.dropout[1](ner_embeddings)
        seq_len = input_ids.shape[1]
        position_sequences = self.get_sentence_positional_feature(input_ids.shape[0], seq_len)

        x_out = []
        for idw in range(config.max_sent):
            position_embeddings = self.position_embeddings(position_sequences[idw].to(self.config.device))
            position_embeddings = self.dropout[2](position_embeddings)
            embeddings = torch.cat((word_embeddings, ner_embeddings, position_embeddings), dim=-1)

            bilstm = self.biLSTM(embeddings)
            bilstm = self.dropout[1](bilstm[0])
            gcn_out = self.gcn(bilstm,
                               input_adj_out,
                               input_adj_inv,
                               input_self,)
            gcn_out = self.dropout[2](gcn_out)
            gcn_out = self.gcn2(gcn_out,
                                input_adj_out,
                                input_adj_inv,
                                input_self,)
            pool_out = self.pooler(gcn_out, idw, input_ners)  # (batch_size, hid_dim)
            pool_out = self.dropout[3](pool_out)
            x_out.append(pool_out)

        x_out = torch.stack(x_out, dim=1)  # batch_size, seq_len, hid_dim
        logits = self.classifier(x_out)

        outputs = (logits,)
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.config.num_class_events)[active_loss]
            activel_labels = labels.view(-1)[active_loss]
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(active_logits, activel_labels)
            outputs += (loss,)

        return outputs   # (logits, loss)

    def params_requires_grad(self):

        return list(filter(lambda p: p.requires_grad, self.parameters()))




if __name__ == "__main__":

    from Event.GCN_2018.utils import load_trimmed_word2vec, load_vocab, encode_window2, load_data_pickle

    print('--> Load vocab: ')
    word2id = load_vocab('data/vocab_word.txt')
    event2id = load_vocab('data/vocab_event_2.txt', False)
    entity2id = load_vocab('data/vocab_ne_2.txt')
    nwords, word2id, id2word, pretrained_embeddings = load_trimmed_word2vec('data/trimmed_word2vec_new.txt')
    # print('Data preparation')
    # word2id.update({'PAD': 0})
    # event2id.update({'PAD': -100})
    # vocab_event = event2id
    # # vocab_event = dict({'O' : 0})
    # # for key in event2id:
    # #     if key[2:] not in vocab_event and key[2:] != '':
    # #         vocab_event.update({key[2:] : len(vocab_event)})
    # # print(vocab_event)
    # for op in ['dev', 'test', 'train']:
    #     print('-->opt: ', op)
    #     words_sents, lab_triggers_sents, entities_sents, dep_sents = load_data_json('data/{}.json'.format(op))
    #     encode_window2(words_sents, lab_triggers_sents, entities_sents, dep_sents, word2id, vocab_event, entity2id,
    #                    window_size=31, save=False, prefix='data/loaddata/{}_'.format(op))

    print('-> Load data')
    train_data = load_data_pickle('data/loaddata/train_', max_sent=31)
    dev_data = load_data_pickle('data/loaddata/dev_', max_sent=31)
    test_data = load_data_pickle('data/loaddata/test_', max_sent=31)

    train_dataset = TensorDataset(train_data[0], train_data[1], train_data[2], train_data[3], train_data[4])
    dev_dataset = TensorDataset(dev_data[0], dev_data[1], dev_data[2], dev_data[3], dev_data[4])
    test_dataset = TensorDataset(test_data[0], test_data[1], test_data[2], test_data[3], test_data[4])

    print('input_ids shape: ', train_data[0].shape)
    print('adj_out matrix shape: ', train_data[1].shape)

    print('-> Build model')
    config = Config()
    config.set_seed(150)
    config.nstep_logging = 50
    config.eval_batch_size = 128
    config.batch_size = 50
    config.learning_rate = 5e-3
    config.num_epoch = 60
    config.weight_decay = 1e-4
    config.warmup_steps = 2000
    config.window_size = 15
    config.fine_tune = True
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model = EDModel2(config, torch.tensor(pretrained_embeddings, dtype=torch.float32))
    model.to(config.device)

    optimizer = optim.Adam(model.params_requires_grad(),
                           weight_decay=config.weight_decay,
                           lr=config.learning_rate,
                           eps=config.adam_eps)
    print(model)

    global_steps = 0.
    f1_best = 0.
    f1_test = 0.
    logging_loss, tr_loss = 0., 0.
    epoch_improve = 0.
    restart_used = 0
    model_name = 'model_gcn_2018.ckpt'
    log_name = 'log_gcn2018.txt'
    tensorboard_name = 'model_1.ckpt'
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)
    total_steps = len(train_loader) * config.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=total_steps)
    tb_writer = SummaryWriter(os.path.join(config.output_dir, tensorboard_name))
    identity_matrix = torch.eye(config.max_sent).unsqueeze(0)

    print('-> Start training process')
    print('nepoch: ', config.num_epoch)
    print('total step: ', total_steps)
    print('step per epoch: ', len(train_loader))

    for ep in range(config.num_epoch):
        train_iterator = tqdm(train_loader)
        for step, batch in enumerate(train_iterator):
            global_steps += 1
            model.train()
            model.zero_grad()
            batch = tuple(t.to(config.device) for t in batch)
            identity_matrix_batch = identity_matrix.repeat(batch[1].shape[0], 1, 1).to(config.device)

            inputs = {"input_ids": batch[0],
                      "input_adj_out": batch[1],
                      "input_adj_inv": batch[2],
                      "input_self": identity_matrix_batch,
                      "input_ners": batch[3],
                      "labels": batch[4]}

            _, loss = model(**inputs)
            tr_loss += loss.item()
            loss.backward()
            # train_iterator.set_description("Epoch {}/{}(lr = {:.10f})-l={:.3f}".format(int(ep), int(config.num_epoch), optimizer.param_groups[0]['lr'], loss.item()))
            torch.nn.utils.clip_grad_norm_(model.params_requires_grad(), 1)
            optimizer.step()
            # if global_steps % 100 == 0:
            scheduler.step()

            if config.nstep_logging > 0 and (
                    global_steps % config.nstep_logging == 0 or step == len(train_iterator) - 1):
                print('lr = {}\n'.format(optimizer.param_groups[0]['lr']))
                # print(loss)

                # print('check', ep, global_steps)
                results = evaluate(config, dev_dataset, model, word2id,
                                   prefix='dev set, step {}/{}'.format(global_steps, ep))
                test_results = evaluate(config, test_dataset, model, word2id,
                                        prefix='test set, step {}/{}'.format(global_steps, ep))
                if test_results['f1'] > f1_test:
                    f1_test = test_results['f1']
                    print('-->Test new best score! f1_test = ', f1_test)
                if results['f1'] > f1_best:
                    f1_best = results['f1']
                    epoch_improve = ep
                    print('--> New best score! f1 = ', f1_best)
                    torch.save(model.state_dict(), os.path.join(config.output_dir, model_name))
                    with open(os.path.join(config.output_dir, log_name), 'a', encoding='utf-8') as f:
                        f.write('Epoch: {:3.0f}, step: {:4.0f} global_step: {:5.0f} (lr= {:.7f})\n\
                                                Results: P= {:.4f} - R= {:.4f} - F= {:.4f} \n \t--=>>>New best score!\n'.format(
                            ep, step, global_steps, optimizer.param_groups[0]['lr'],
                            results['precision'],
                            results['recall'],
                            results['f1']))
                    for key, value in results.items():
                        if key != 'loss':
                            tb_writer.add_scalar("{} score".format(key), value, global_step)
                    tb_writer.add_scalar("learning rate", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalars("loss", {'train_loss': (tr_loss - logging_loss) / args.nstep_logging,
                                                   'dev_loss': results['loss']}, global_step)

                    logging_loss = tr_loss

                else:
                    with open(os.path.join(config.output_dir, log_name), 'a', encoding='utf-8') as f:
                        f.write('Epoch: {:3.0f}, step: {:4.0f} global_step: {:5.0f} (lr= {:.7f})\n\
                                Results: P= {:.4f} - R= {:.4f} - F= {:.4f}\n'.format(ep, step, global_steps,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     results['precision'],
                                                                                     results['recall'],
                                                                                     results['f1']))

                    if ep - epoch_improve > 10:
                        if restart_used > config.max_restart:
                            print('Restarting model is run out')
                            break
                        else:
                            restart_used += 1
                            print('--->>>RELOAD MODEL from epoch {}'.format(epoch_improve))
                            with open(os.path.join(config.output_dir, log_name), 'a', encoding='utf-8') as f:
                                f.write('---->>>>RELOAD MODEL FROM EPOCH {}\n'.format(epoch_improve))
                            model.load_state_dict(torch.load(os.path.join(config.output_dir, model_name)))
                            epoch_improve = ep
                            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                        num_warmup_steps=config.warmup_steps * (
                                                                                    total_steps - global_steps) / total_steps,
                                                                        num_training_steps=total_steps - global_steps)

        # if (ep+1) % 5 == 0:
        #     output.clear()

    print('-->FINAL TEST')
    test_results = evaluate(config, test_dataset, model, word2id, prefix='test set- final test')





