import torch
import torch.nn.functional as F

from icecream import ic


# Soft aggregation from STM
def aggregate(prob, dim, return_logits=False):
    # Questions I'd like to answer:
    # dim = ? 0 I suppose, taking the object class
    # Range of prob? I think it's output is between either [0,1] or [0,+inf] due to the exponential, depending if working with prob or logits
    # torch.log convert the probability back to logits
    # Softmax convert the logits back to a probability
    txt = "Soft Aggregation of the Predicted Probability"
    # ic(txt)
    # prob.shape [#obj, H, W] 
    # ic(prob.shape, dim, return_logits)
    new_prob = torch.cat([
        torch.prod(1-prob, dim=dim, keepdim=True),
        prob
    ], dim).clamp(1e-7, 1-1e-7)
    # ic(new_prob.shape)
    logits = torch.log((new_prob /(1-new_prob)))
    # ic(logits.shape)
    prob = F.softmax(logits, dim=dim)
    # ic(prob.shape)
    # prob.shape [#number of objkects +1 background, H, W] 

    if return_logits:
        return logits, prob
    else:
        return prob
