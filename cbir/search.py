import torch
import torch.nn.functional as F


def batched_coattention_search(V_q: torch.Tensor, db_tensor: torch.Tensor, T: float = 10) -> torch.Tensor:
    """
    Run co-attention search on the database images.

    Args:
        V_q (torch.Tensor) - The query vector
        db_tensor (torch.Tensor) - All cached database vectors stacked

    Returns:
        scores - Similarity scores for the whole database
    """
    Num_DB = db_tensor.shape[0]
    V_q_expanded = V_q.expand(Num_DB, -1).unsqueeze(1)

    sim_scores = F.cosine_similarity(V_q_expanded, db_tensor, dim=-1)
    a = F.softmax(sim_scores * T, dim=-1)
    V_c = torch.sum(a.unsqueeze(-1) * db_tensor, dim=1)
    final_scores = F.cosine_similarity(V_q.expand(Num_DB, -1), V_c, dim=-1)

    return final_scores
