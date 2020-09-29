from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
import torch
from copy import deepcopy

@log_args(but_as=Learner.__init__)
class LT_TabularLearner(Learner):
    
    # Copied from tabular\learner.py
    
    "`Learner` for tabular data"
    def predict(self, row):
        "Predict on a Pandas Series"
        dl = self.dls.test_dl(row.to_frame().T)
        dl.dataset.conts = dl.dataset.conts.astype(np.float32)
        inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
        b = (*tuplify(inp),*tuplify(dec_preds))
        full_dec = self.dls.decode(b)
        return full_dec,dec_preds[0],preds[0]

# Cell
class LT_TabularModel(Module):
    
    # Copied from tabular/model.py
    
    "Basic model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True)):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)
        
        # custom code below
        # save the initial weights of the model so we can restore them after training
        self.LT_old_weights = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                if isinstance(self.layers[i][j], torch.nn.Linear):
                    self.LT_old_weights.append(deepcopy(self.layers[i][j]))
                    
    def LT_prune_layers(self, p=0.4):
        for layer in self.layers:
            for l in layer:
                if isinstance(l, torch.nn.Linear):
                    torch.prune.l1_unstructured(module, name='weight', amount=p)
                    
    def LT_restore_weights(self):
        k = 0
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                if isinstance(self.layers[i][j], torch.nn.Linear):
                    with torch.no_grad():
                        self.layers[i][j].weight.copy_(self.LT_old_weights[k].weight)
                    k += 1
                elif isinstance(self.layers[i][j], torch.nn.BatchNorm1d):
                    self.layers[i][j] = torch.nn.BatchNorm1d(self.layers[i][j].num_features)
    
    def LT_dump_weights(self):
        ret = ""
        for layer in self.layers:
            for i in range(len(layer)):
                if isinstance(layer[i], torch.nn.Linear):
                    ret += str(layer[i].weight)
        return ret
        
    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)
    

@log_args(to_return=True, but_as=Learner.__init__)
@delegates(Learner.__init__)
def LT_tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, **kwargs):
    
    # Copied from tabular\learner.py
    
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None: config = tabular_config()
    if layers is None: layers = [200,100]
    to = dls.train_ds
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = LT_TabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, **config)
    return LT_TabularLearner(dls, model, **kwargs)
