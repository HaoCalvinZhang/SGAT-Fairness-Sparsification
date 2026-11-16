import numpy as np
import torch
import torch.nn.functional as F
from sgat_fair.model import SGAT,build_directed_edges
from sgat_fair.metrics import soft_dp_loss,classification_scores


def toy_model():
    torch.manual_seed(7)
    x=torch.randn(6,2)
    u=np.array([0,0,1,2,3,4],dtype=np.int64)
    v=np.array([1,2,3,4,5,5],dtype=np.int64)
    edges=build_directed_edges(u,v,6,torch.device('cpu'))
    model=SGAT(2,hidden=5,num_heads=2,num_layers=1,
               input_dropout=0,attention_dropout=0, bias_l0=0)
    return model,x,edges


def test_l0_sum_and_self_loops():
    model,x,edges=toy_model()
    model.eval()
    out,g=model(x,edges,return_gates=True)
    assert out.shape==(6,2)
    assert torch.all(g.gate[-6:]==1)
    assert torch.allclose(g.l0_sum,g.nonzero_probabilities.sum(),atol=1e-6)
    # With all loops forced to 1 and no attention dropout, each dst normalized sums to 1.
    dst=edges[1]
    totals=torch.zeros(6).index_add(0,dst,g.normalized_weights)
    assert torch.allclose(totals,torch.ones(6),atol=1e-6)
    assert g.expected_keep_pct.ndim==0


def test_original_gate_formula_temperature():
    model,_,_=toy_model()
    x=torch.tensor([-5.0,-1.0,0.0,1.0])
    gate=model.gate_weights(x,'deterministic')
    want=(torch.sigmoid(x/0.66)*1.2-0.1).clamp(0,1)
    assert torch.allclose(gate,want)
    p=model.l0_probability(x)
    want_p=torch.sigmoid(x-0.66*np.log(.1/1.1))
    assert torch.allclose(p,want_p)
    assert set(torch.unique(model.gate_weights(x,'binary')).tolist()).issubset({0.0,1.0})


def test_fairness_and_sparsity_gradients_reach_attention_parameters():
    model,x,edges=toy_model()
    s=torch.tensor([0,0,0,1,1,1])
    y=torch.tensor([0,1,1,0,1,0])
    ids=torch.arange(6)
    model.train()
    logits,g=model(x,edges,return_gates=True)
    loss=F.cross_entropy(logits[ids],y)+0.0001*g.l0_sum+5*soft_dp_loss(logits,s,ids)
    loss.backward()
    assert model.layers[0].att_src.grad is not None
    assert torch.isfinite(model.layers[0].att_src.grad).all()
    assert model.layers[0].att_src.grad.abs().sum()>0
    assert model.layers[0].bias_l0.grad is not None


def test_beta_changes_fairness_term_gradients():
    model,x,edges=toy_model()
    model.eval()
    s=torch.tensor([0,0,0,1,1,1]); ids=torch.arange(6)
    logits,_=model(x,edges,return_gates=True)
    dp=soft_dp_loss(logits,s,ids)
    g=torch.autograd.grad(dp,model.layers[0].att_src,retain_graph=True)[0]
    g2=torch.autograd.grad(10*dp,model.layers[0].att_src)[0]
    assert torch.allclose(g2,10*g,atol=1e-5)


def test_metric_bounds():
    logits=torch.tensor([[5.,0.],[0.,5.],[5.,0.],[0.,5.],[0.,5.],[5.,0.]])
    ys=torch.tensor([0,1,1,0,1,0]); s=torch.tensor([0,0,0,1,1,1]); ids=torch.arange(6)
    metrics=classification_scores(logits,ys,s,ids)
    assert 0<=metrics['accuracy_pct']<=100
    assert 0<=metrics['dp_gap']<=1
    assert 0<=metrics['eo_gap']<=1
