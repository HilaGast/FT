import numpy as np
import random
import networkx as nx

def find_neighbours(graph, node, path_list):
    '''Find all the neighbours of a node in a graph'''
    neighbours = list(graph.neighbors(node))
    for n in neighbours:
        if n in path_list:
            neighbours.remove(n)

    return neighbours

def choose_neighbour_weighted_by_edge_weight(graph, node, neighbours):

    weights = [graph[node][neighbour] for neighbour in neighbours]
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    chosen_neighbour = random.choices(neighbours, cum_weights=weights)

    return chosen_neighbour[0]

def add_chosen_neighbour_to_path_list(path_list, node):

    path_list.append(node)
    return path_list

def add_edge_weight_to_path_weight(path_weight, graph, node, chosen_neighbour):

        path_weight += graph[node][chosen_neighbour]
        return path_weight

def random_walk(graph_num, graph_weights, start_node, max_steps=50, max_path_weight=1000):

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
            node = chosen_neighbour
            steps += 1

        return path_list, path_weight