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


def batched_coattention_search_pruned(V_q: torch.Tensor, db_clusters: torch.Tensor, db_centroids: torch.Tensor, db_radii: torch.Tensor, T: float = 10, keep_top_n: int = 100) -> tuple:
    """
    Fast search using Macro-Pruning and Einstein Summation
    """
    V_q_norm = F.normalize(V_q, p=2, dim=-1)
    
    centroid_sims = torch.mm(db_centroids, V_q_norm.T).squeeze(-1)
    max_possible_scores = centroid_sims + db_radii 
    
    _, candidate_indices = torch.topk(max_possible_scores, k=min(keep_top_n, db_clusters.shape[0]))
    surviving_clusters = db_clusters[candidate_indices]
    
    surviving_clusters_norm = F.normalize(surviving_clusters, p=2, dim=-1)
    
    sim_scores = torch.einsum('id,nkd->nk', V_q_norm, surviving_clusters_norm)
    a = F.softmax(sim_scores * T, dim=-1)
    
    V_c = torch.einsum('nk,nkd->nd', a, surviving_clusters)
    V_c_norm = F.normalize(V_c, p=2, dim=-1)
    
    final_scores = torch.mm(V_c_norm, V_q_norm.T).squeeze(-1)

    return candidate_indices, final_scores
