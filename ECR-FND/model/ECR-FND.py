import torch
import torch.nn as nn
import torch.nn.functional as F
from .trm import *
import pandas as pd
import json
from .attention import *
from utils.loss_function import *

class EVDL(torch.nn.Module):
    def __init__(self):
        super(EVDL, self).__init__()

        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
        self.input_visual_frames = 83

        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))

        self.mlp_img = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1))

        self.mlp_descriptor = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1))

        self.co_attention_tv = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128, visual_len=self.input_visual_frames, sen_len=512, fea_v=128, fea_s=128, pos=False)

        self.content_classifier = nn.Linear(128, 2)

        self.score_fc = nn.Linear(128, 1)
        self.d_common = 128

        self.loss_fn = EVDLLoss(margin=0.2, tau=0.07)

    def forward(self, **kwargs):


        raw_t_fea_semantic = self.mlp_text_semantic(kwargs['all_phrase_semantic_fea'])

        raw_v_fea = self.mlp_img(kwargs['raw_visual_frames'])


        descriptor_enhanced = self.mlp_descriptor(kwargs['all_descriptor_fea']).unsqueeze(1)
        enhanced_text = torch.cat((raw_t_fea_semantic, descriptor_enhanced), dim=1)  # (B, L_text+1, 128)
        content_v, _ = self.co_attention_tv(v=raw_v_fea, s=raw_t_fea_semantic, v_len=raw_v_fea.shape[1], s_len=enhanced_text.shape[1])

        global_text = torch.mean(enhanced_text, dim=1)


        consistency_cos = F.cosine_similarity(content_v, global_text.unsqueeze(1).expand_as(content_v),
                                              dim=-1)

        lin_score = self.score_fc(content_v).squeeze(-1)
        scaled_score = lin_score / math.sqrt(self.d_common)

        consistency_scores = scaled_score * consistency_cos

        L = consistency_scores.shape[1]
        mask = torch.ones_like(consistency_scores, dtype=torch.bool)
        min_indices = torch.argmin(consistency_scores, dim=1)
        for i in range(B):
            mask[i, min_indices[i]] = False
        content_v_filtered = content_v[mask].view(B, L - 1, -1)
        consistency_scores_filtered = consistency_scores[mask].view(B, L - 1)

        global_video = torch.mean(content_v_filtered, dim=1)

        loss_total, loss_mr, loss_info = self.loss_fn(consistency_scores_filtered, global_text, global_video)

        fusion_semantic_fea = torch.mean(torch.cat((content_v_filtered, enhanced_text), dim=1), dim=1)
        output_EVDL = self.content_classifier(fusion_semantic_fea)

        return output_EVDL, loss_mr, loss_info

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PosEncoding_fix(nn.Module):
    def __init__(self,  d_word_vec):
        super(PosEncoding_fix, self).__init__()
        self.d_word_vec=d_word_vec
        self.w_k=np.array([1/(np.power(10000,2*(i//2)/d_word_vec)) for i in range(d_word_vec)])

    def forward(self, inputs):
        
        pos_embs=[]
        for pos in inputs:
            pos_emb=torch.tensor([self.w_k[i]*pos.cpu() for i in range(self.d_word_vec)])
            if pos !=0:
                pos_emb[0::2]=np.sin(pos_emb[0::2])
                pos_emb[1::2]=np.cos(pos_emb[1::2])
                pos_embs.append(pos_emb)
            else:
                pos_embs.append(torch.zeros(self.d_word_vec))
        pos_embs=torch.stack(pos_embs)
        return pos_embs.cuda()

class DurationEncoding(nn.Module):
    def __init__(self,dim,dataset):
        super(DurationEncoding,self).__init__()
        if dataset=='fakett':
            #'./fea/fakett/fakett_segment_duration.json' record the duration of each clip(segment) for each video
            with open('./datasets/fea/fakett/fakett_segment_duration.json', 'r') as json_file:
                seg_dura_info=json.load(json_file)
        elif dataset=='fakesv':
            #'./fea/fakesv/fakesv_segment_duration.json' record the duration of each clip(segment) for each video
            with open('./datasets/fea/fakesv/fakesv_segment_duration.json', 'r') as json_file:
                seg_dura_info=json.load(json_file)
        
        self.all_seg_duration=seg_dura_info['all_seg_duration']
        self.all_seg_dura_ratio=seg_dura_info['all_seg_dura_ratio']
        self.absolute_bin_edges=torch.quantile(torch.tensor(self.all_seg_duration).to(torch.float64),torch.range(0,1,0.01).to(torch.float64)).cuda()
        self.relative_bin_edges=torch.quantile(torch.tensor( self.all_seg_dura_ratio).to(torch.float64),torch.range(0,1,0.02).to(torch.float64)).cuda()
        self.ab_duration_embed=torch.nn.Embedding(101,dim)
        self.re_duration_embed=torch.nn.Embedding(51,dim)

        

        self.ocr_all_seg_duration=seg_dura_info['ocr_all_seg_duration']
        self.ocr_all_seg_dura_ratio=seg_dura_info['ocr_all_seg_dura_ratio']
        self.ocr_absolute_bin_edges=torch.quantile(torch.tensor(self.ocr_all_seg_duration).to(torch.float64),torch.range(0,1,0.01).to(torch.float64)).cuda()
        self.ocr_relative_bin_edges=torch.quantile(torch.tensor( self.ocr_all_seg_dura_ratio).to(torch.float64),torch.range(0,1,0.02).to(torch.float64)).cuda()
        self.ocr_ab_duration_embed=torch.nn.Embedding(101,dim)
        self.ocr_re_duration_embed=torch.nn.Embedding(51,dim)

        self.result_dim=dim

    def forward(self,time_value,attribute):
        all_segs_embedding=[]
        if attribute=='natural_ab':
            for dv in time_value:
                # bucket_indice=torch.searchsorted(self.absolute_bin_edges, torch.tensor(dv,dtype=torch.float64))
                bucket_indice = torch.searchsorted(self.absolute_bin_edges, dv.clone().detach().to(torch.float64))
                dura_embedding=self.ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute=='natural_re':
            for dv in time_value:
                # bucket_indice=torch.searchsorted(self.relative_bin_edges, torch.tensor(dv,dtype=torch.float64))
                bucket_indice = torch.searchsorted(self.relative_bin_edges, dv.clone().detach().to(torch.float64))
                dura_embedding=self.re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute=='ocr_ab':
            for dv in time_value:
                # bucket_indice=torch.searchsorted(self.ocr_absolute_bin_edges, torch.tensor(dv,dtype=torch.float64))
                bucket_indice = torch.searchsorted(self.ocr_absolute_bin_edges, dv.clone().detach().to(torch.float64))
                dura_embedding=self.ocr_ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
                
        elif attribute=='ocr_re':
            for dv in time_value:
                # bucket_indice=torch.searchsorted(self.ocr_relative_bin_edges, torch.tensor(dv,dtype=torch.float64))
                bucket_indice = torch.searchsorted(self.ocr_relative_bin_edges, dv.clone().detach().to(torch.float64))
                dura_embedding=self.ocr_re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
                

        if len(all_segs_embedding)==0:
            return torch.zeros((1,self.result_dim)).cuda() 
        return torch.stack(all_segs_embedding,dim=0).cuda() 


def get_dura_info_visual(segs,fps,total_frame):
    duration_frames=[]
    duration_time=[]
    for seg in segs:
        if seg[0]==-1 and seg[1]==-1:
            continue
        if seg[0]==0 and seg[1]==0:
            continue
        else:
            duration_frames.append(seg[1]-seg[0]+1)
            duration_time.append((seg[1]-seg[0]+1)/fps)
    duration_ratio=[min(dura/total_frame,1) for dura in duration_frames]
    return torch.tensor(duration_time).cuda(),torch.tensor(duration_ratio).cuda()



class ATCM(torch.nn.Module):
    def __init__(self, dataset):
        super(ATCM, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768

        self.encoded_audio_semantic_fea_dim = 512
        self.encoded_emo_fea_dim = 128
        self.mlp_text_emo = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(),
                                               nn.Dropout(0.1))
        self.mlp_audio_semantic = nn.Sequential(nn.Linear(self.encoded_audio_semantic_fea_dim, 128), nn.ReLU(),
                                                nn.Dropout(0.1))

        self.mlp_audio = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.ReLU(), nn.Dropout(0.1))

        self.trm_text_emo = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        self.audio_trm_emo = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)

        self.co_attention_ta_semantic = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,
                                                     text_len=self.encoded_text_semantic_fea_dim, audio_len=512,
                                                     fea_v=128, fea_s=128,
                                                     pos=False)

        self.co_attention_ta_emo = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,
                                                text_len=self.encoded_emo_fea_dim, audio_len=512, fea_v=128, fea_s=128,
                                                pos=False)

        self.co_attention_at_emo = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,
                                                audio_len=self.encoded_emo_fea_dim, sen_len=512, fea_v=128, fea_s=128,
                                                pos=False)

        self.content_classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))

    def forward(self, **kwargs):

        text_semantic = kwargs['all_phrase_semantic_fea']
        audio_semantic = kwargs['all_audio_semantic_fea']
        audio_emo = kwargs['raw_audio_emo']
        descriptor_semantic = self.mlp_descriptor(kwargs['all_descriptor_fea']).unsqueeze(1)

        H_text = self.mlp_text_semantic(torch.cat((text_semantic, descriptor_semantic), dim=1))
        H_audio = self.mlp_audio_semantic(audio_semantic)


        H_audio_enhanced, _ = self.co_attention_ta_semantic(H_audio, H_text, H_audio)
        H_text_enhanced, _ = self.co_attention_ta_semantic(H_text, H_audio, H_text)


        consistency_matrix = torch.matmul(H_audio_enhanced, H_text.transpose(1, 2))
        R_con = torch.softmax(consistency_matrix.sum(dim=2), dim=-1)
        R_incon = 1 - R_con.unsqueeze(-1)


        H_audio_prime = R_incon * H_audio_enhanced

        # ===== 情感特征处理 =====
        text_emo = self.mlp_text_emo(text_emo)
        audio_emo = self.mlp_audio(audio_emo)


        text_emo = self.trm_text_emo(text_emo)
        audio_emo = self.audio_trm_emo(audio_emo)

        # 跨模态情感交互
        E_audio, E_text = self.co_attention_ta_emo(audio_emo, text_emo, audio_emo)

        audio_semantic_global = torch.mean(H_audio_prime, dim=1).unsqueeze(1)
        text_semantic_global = torch.mean(H_text, dim=1).unsqueeze(1)

        audio_emo_global = torch.mean(E_audio, dim=1).unsqueeze(1)
        text_emo_global = torch.mean(E_text, dim=1).unsqueeze(1)

        combined_features = torch.mean(torch.cat([
            audio_semantic_global,
            audio_emo_global,
            text_semantic_global,
            text_emo_global
        ], dim=1))


        output_ATCM= self.content_classifier(combined_features)

        return output_ATCM


class ECR-FND(torch.nn.Module):

    def __init__(self,dataset):
        super(ECR-FND,self).__init__()
        self.EVDL_branch=EVDL(dataset=dataset)
        self.ATCM_branch=ATCM(dataset=dataset)

        self.tanh = nn.Tanh()

    def forward(self,  **kwargs):
        output_EVDL, loss_mr, loss_info=self.EVDL_branch(**kwargs)
        output_ATCM=self.ATCM_branch(**kwargs)
        output=output_EVDL*self.tanh(output_ATCM)

        return output, loss_mr, loss_info


