import numpy as np
import random
import networkx as nx

def find_neighbours(graph, node, path_list):
    '''Find all the neighbours of a node in a graph'''
    neighbours = list(graph.neighbors(node))
    new_neighbours = [neighbour for neighbour in neighbours if neighbour not in path_list]

    return new_neighbours

def choose_neighbour_weighted_by_edge_weight(graph, node, neighbours):
    '''Choose a neighbour of a node in a graph, weighted by the edge weight'''
    weights = [graph[node][neighbour]['weight'] for neighbour in neighbours]
    weights = np.array(weights)
    weights = weights / np.nanmax(weights)
    chosen_neighbour = random.choices(neighbours, weights=weights)

    return chosen_neighbour[0]

def add_chosen_neighbour_to_path_list(path_list, node):
    '''Add a chosen neighbour to the path list'''
    path_list.append(node)
    return path_list

def add_edge_weight_to_path_weight(path_weight, graph, node, chosen_neighbour):
    '''Add the weight of an edge to the path weight'''
    path_weight += graph[node][chosen_neighbour]['weight']
    return path_weight

def add_current_path_weight_to_node_dict(node_dict, node, path_weight):
    '''Add the current path weight to the node dictionary'''
    node_dict[node].append(path_weight)
    return node_dict

def random_walk(graph_num, graph_weights, start_node, max_steps=100, max_path_weight=10):
    '''Random walk on a graph, with a maximum number of steps and a maximum path weight'''
    node_dict = {node: [np.nan] for node in graph_num.nodes()}
    path_list = [start_node]
    path_weight = 0
    node = start_node
    steps = 0
    while steps < max_steps and path_weight < max_path_weight:
        neighbours = find_neighbours(graph_num, node, path_list)
        if len(neighbours) == 0:
            break
        chosen_neighbour = choose_neighbour_weighted_by_edge_weight(graph_num, node, neighbours)
        path_list = add_chosen_neighbour_to_path_list(path_list, chosen_neighbour)
        path_weight = add_edge_weight_to_path_weight(path_weight, graph_weights, node, chosen_neighbour)
        if path_weight > max_path_weight:
            break
        node = chosen_neighbour
        steps += 1
        node_dict = add_current_path_weight_to_node_dict(node_dict, node, path_weight)

    return node_dict