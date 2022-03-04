import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import math

from IPython import embed
import numpy as np
from torch.nn.parameter import Parameter

INF = 1e8

class DocREModel(nn.Module):
    def __init__(self, config, model, ablation='atlop', max_sen_num=25, emb_size=768, block_size=64, num_labels=-1, att_size=128, b_hid_size=194):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        output_size = config.num_labels

        self.max_sen_num = max_sen_num
        self.ablation = ablation

        if self.ablation in ['eider', 'eider_rule']:
            self.b_hid_size = b_hid_size

            self.sr_bilinear = nn.Linear(config.hidden_size * block_size, config.num_labels)

        extractor_input_size = 2 * config.hidden_size

        self.head_extractor = nn.Linear(extractor_input_size, emb_size)
        self.tail_extractor = nn.Linear(extractor_input_size, emb_size)

        self.bilinear = nn.Linear(emb_size * block_size, output_size)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.num_rels = output_size

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention, sentence_cls = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention, sentence_cls

    def get_hrt(self, sequence_output, attention, entity_pos, hts, sen_poss=None):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []

        sss, sss_simple = [], []

        for i in range(len(entity_pos)): # for i in batch_size
            entity_embs, entity_atts = [], []
            for eid,e in enumerate(entity_pos[i]):

                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        m_num = len(e_emb)
                        e_emb = torch.stack(e_emb, dim=0)
                        e_emb = torch.logsumexp(e_emb, dim=0) # take the average here

                        e_att = torch.stack(e_att, dim=0)
                        e_att = e_att.mean(0) # (h, seq_len) take average on att anyway..
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            if sen_poss is not None:
                # logsumexp:
                ss = [torch.logsumexp(sequence_output[i, sen_pos[0]+offset:sen_pos[1]+offset, :], dim=0).unsqueeze(0) for sen_pos in sen_poss[i] if sen_pos[0]+offset < c]
                ss = torch.cat(ss, dim=0).to(sequence_output.device) # [sen_num, emb_size]
                # padding ss to [max_sen_num, emb_size]
                pad = nn.ZeroPad2d((0, 0, 0, self.max_sen_num - ss.shape[0]))
                ss = pad(ss)

                p_num = len(hts[i])
                sss.append(ss.unsqueeze(0).expand(p_num, self.max_sen_num, self.hidden_size))
                sss_simple.append(ss.unsqueeze(0))

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d] OR [n_e, k, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device) # (n_p, 2)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0]) # [n_p, d] OR [n_p, k, d]
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1]) # [n_p, d] OR [n_p, k, d]

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0]) # (n_p, h, seq_len)
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1]) # (n_p, h, seq_len)
            ht_att = (h_att * t_att).mean(1) # average on all heads # (n_p, seq_len)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5) # average over seq_len

            rs = contract("ld,rl->rd", sequence_output[i], ht_att) # (seq_len, d), (n_p, seq_len) -> [n_p, d]

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0) # [bs * n_p, d] OR [bs * n_p, num_labels, d]
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)

        if len(sss)>0:
            sss = torch.cat(sss, dim=0) # [bs * n_p, num_sents, d]
            sss_simple = torch.cat(sss_simple, dim=0) # [bs, num_sents, d]

        return hss, rss, tss, sss, sss_simple

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                sen_labels=None,
                sen_pos=None,
                return_senatt=False,
                return_score=False,
                ):

        sequence_output, attention, sentence_cls = self.encode(input_ids, attention_mask)

        max_sen_num = self.max_sen_num
        if torch.isnan(sequence_output).any():
            print('bert nan')
            embed()

        hs, rs, ts, \
        ss, ss_simple = self.get_hrt(sequence_output, attention, entity_pos, hts, \
                                                sen_poss=sen_pos)

        hs = torch.cat([hs, rs], dim=-1) # hs, rs: input: [bs * n_p, d] OR [bs * n_p, num_labels, d]
        ts = torch.cat([ts, rs], dim=-1) # output: [bs * n_p, emb_size] OR [bs * n_p, num_labels, emb_size]

        hs = torch.tanh(torch.tanh(self.head_extractor(hs))) # hs, rs: input: [bs * n_p, d] OR [bs * n_p, num_labels, d]
        ts = torch.tanh(self.tail_extractor(ts)) # output: [bs * n_p, emb_size] OR [bs * n_p, num_labels, emb_size]

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        # b1: [bs', bl_num, bl] -> [bs', bl_num, bl, 1]
        # b2: [bs', bl_num, 1, bl]
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size) # bl: [bs * n_p, emb_size * block_size]

        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)

        if return_score:
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output = output + (scores_topk[0], scores_topk[1], )

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]

            if sen_labels is not None:
                num_idx_used = [ len(torch.nonzero(label[:,1:].sum(dim=-1))) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)

            loss = self.loss_fnt(logits.float(), labels.float())

            if sen_labels is not None: # 'eider'
                s_labels = [torch.tensor(s_label) for s_label in sen_labels] # sen_labels: list of 2d lists
                s_labels = torch.cat(s_labels, dim=0).to(ss) # [ps, max_sen_num]
                idx_used = torch.nonzero(labels[:,1:].sum(dim=-1)).view(-1)

                # pred important sent for each (h,t,r)
                # hs, ts: [bs, d] -> [bs', sents_num, d]
                # ss: [bs, sents_num, d]
                hs = hs[idx_used]
                ts = ts[idx_used]
                ss = ss[idx_used]
                rs = rs[idx_used]

                s1 = [ss_simple[i].unsqueeze(0).expand(num_idx_used[i], self.max_sen_num, self.hidden_size) for i in range(ss_simple.shape[0])]
                s1 = torch.cat(s1, dim=0) # (bs', sents_num, hid)
                s1 = s1.view(-1, self.max_sen_num, self.hidden_size // self.block_size, self.block_size) # (bs', sents_num, #bl, bl_size)
                r2 = rs.view(-1, self.hidden_size // self.block_size, self.block_size) # (bs', #bl, bl_size)
                bl_sr = (s1.unsqueeze(4) * r2.unsqueeze(1).unsqueeze(3)).view(-1, self.max_sen_num, self.hidden_size * self.block_size)
                # s1 -> (bs', sents_num, #bl, bl_size, 1); r2 -> (bs', 1, #bl, 1, bl_size)
                s_logits = self.sr_bilinear(bl_sr) # [bs, sents_num, num_labels]

                s_logits = torch.max(s_logits, dim=-1)[0].view(-1, max_sen_num) # choose the highest prob
                evi_loss = F.binary_cross_entropy_with_logits(s_logits.float(), s_labels.float())
                loss = loss + 0.1*evi_loss
                # loss = evi_loss

                if torch.isnan(loss):
                    print('loss nan')
                    embed()

                if return_senatt:
                    s_output = self.loss_fnt.get_label(s_logits, num_labels=self.num_labels).view(-1, self.max_sen_num, self.num_rels)
                    output = output + (s_output, )

            output = (loss.to(sequence_output),) + output

        elif return_senatt:
            num_idx_used = [len(hts_i) for hts_i in hts]

            s1 = ss.view(-1, self.max_sen_num, self.hidden_size // self.block_size, self.block_size) # (bs', sents_num, #bl, bl_size)
            r2 = rs.view(-1, self.hidden_size // self.block_size, self.block_size) # (bs', #bl, bl_size)
            bl_sr = (s1.unsqueeze(4) * r2.unsqueeze(1).unsqueeze(3)).view(-1, self.max_sen_num, self.hidden_size * self.block_size)
            # s1 -> (bs', sents_num, #bl, bl_size, 1); r2 -> (bs', 1, #bl, 1, bl_size)
            s_logits = self.sr_bilinear(bl_sr) # [bs, sents_num, num_labels]

            s_logits = torch.max(s_logits, dim=-1)[0].view(-1, max_sen_num) # choose the highest prob, [bs', sents_num]
            s_output = s_logits > 0
            output = output + (s_output, )

        return output
