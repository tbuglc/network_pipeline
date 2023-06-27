import networkx as nx
from structure import load_accorderie_network




def duree_to_int(duree_str):
    ret = 0
    pz = duree_str.split(":")
    ret += float(pz[0])
    ret += float(pz[1]) / 60 * 100
    return ret


# average indeg / (indeg + outdeg).  Less than 0.5 => outdeg bias, higher => indeg bias
def get_avg_in_out_degree(g):
    
    if len(g.vs) == 0:
        return -1
    
    ratio_sum = 0
    nb_isolated = 0
    
    for v in g.vs:
        indeg = g.degree(v, mode='in')
        outdeg = g.degree(v, mode='out')
        
        if outdeg == 0 and indeg == 0:
            nb_isolated += 1
        else:
            ratio_sum += (indeg / (outdeg + indeg))
    
    return ratio_sum / (len(g.vs) - nb_isolated) 







# average duree_in / (duree_in + duree_out).  Less than 0.5 => outdeg bias, higher => indeg bias
def get_avg_weighted_in_out_degree(g, field_name='duree'):
    
    if len(g.vs) == 0:
        return -1
    
    ratio_sum = 0
    nb_isolated = 0
    
    for v in g.vs:
        weight_in = 0
        for e in g.es[g.incident(v, mode='in')]:
            weight_in += duree_to_int(e['duree'])
        
        weight_out = 0
        for e in g.es[g.incident(v, mode='out')]:
            weight_out += duree_to_int(e['duree'])
        
        
        
        if weight_in == 0 and weight_out == 0:
            nb_isolated += 1
        else:
            ratio_sum += (weight_in / (weight_in + weight_out))
        
    
    return ratio_sum / (len(g.vs) - nb_isolated) 










# average max of indeg / (indeg + outdeg) or 1 - that qty.  Minimum is 0.5, closer to 1 => quite disbalanced
def get_avg_in_out_disbalance(g):
    
    if len(g.vs) == 0:
        return -1
    
    disbalance_sum = 0
    nb_isolated = 0
    for v in g.vs:
        indeg = g.degree(v, mode='in')
        outdeg = g.degree(v, mode='out')
        
        if outdeg == 0 and indeg == 0:
            nb_isolated += 1
        else:
            disbalance_sum += max(indeg / (indeg + outdeg), 1 - indeg/(indeg + outdeg))
    
    return disbalance_sum / (len(g.vs) - nb_isolated) 






# ratio of unique edges / edges.  Under 1 => edges are repeated
def get_unique_edges_vs_total(g):
    
    if len(g.es) == 0:
        return -1
    
    nb_edges = len(g.es)
    
    unique_edges = set()
    all_edges = list()
    
    for e in g.es:
        et = e.tuple
        if et not in unique_edges:
            unique_edges.add(et)
            
        all_edges.append(et)

    #all_edges.sort()
    #print(all_edges)

            
    return len(unique_edges) / nb_edges















g = load_accorderie_network('./data/sherbrooke/data/members.csv', './data/sherbrooke/data/transactions.csv')




avg_inout = get_avg_in_out_degree(g)
print(f"avg_inout={avg_inout}")


avg_weighted_inout = get_avg_weighted_in_out_degree(g, field_name='duree')
print(f"avg_weighted_inout={avg_weighted_inout}")

avg_disbalance = get_avg_in_out_disbalance(g)
print(f"avg_disbalance={avg_disbalance}")


unique_edge_ratio = get_unique_edges_vs_total(g)
print(f"unique_edge_ratio={unique_edge_ratio}")




