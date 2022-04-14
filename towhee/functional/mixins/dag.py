from uuid import uuid4
from copy import deepcopy

from towhee.hparam import param_scope


OP_EQUIVALENTS = {
    'split': 'nop'
}


class DagMixin:
    """
    Mixin for creating DAGs and their corresponding yamls from a DC
    """

    def __init__(self) -> None:
        super().__init__()
        # Unique id for current operation
        self._id = str(uuid4().hex[:8])
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        # If there is a parent and you werent given a dag, use it and add yourself to its children
        if parent is not None:
            # If you dont want isolated nodes, swap lines below
            # self._dag = deepcopy(parent._dag)
            self._dag = parent._dag
            self._dag[parent._id][2].append(self._id)
        # If first opeartions, you have no parent, so you are a root.
        else:
            self._dag = {'start_op': ('start_op', (), [self._id])}

    @property
    def id(self):
        return self._id

    @property
    def dag(self):
        return self._dag

    def plotable_dag(self):
        new_dict = {}
        for key, value in self._dag.items():
            new_vals = []
            for x in value[2]:
                try:
                    temp = self._dag[x][0] + '_' + x
                    new_vals.append(temp)
                except KeyError:
                    temp = 'end' +  '_' + x
                    new_vals.append(temp)
            new_dict[value[0] + '_' + key] = new_vals
        return new_dict

    

    # def dag_to_yaml(self):
    #     def attach_end(dag):
    #         replace = None
    #         for value in list(dag.values()):
    #             for x in value[2]:
    #                 if x not in dag:
    #                     replace = x
    #                     dag['end_op'] = ('end_op', (), [])
    #                     break
    #         for key, value in dag.items():
    #             try: 
    #                 n = value[2].index(replace)
    #                 value[2][n] = 'end_op'
    #             except:
    #                 pass
    #         return dag
    
    #     dataframes = {}
    #     ops = {}
    #     dag = attach_end(deepcopy(self._dag))
    #     df_counter = 0
    #     print(dag)
    #     for key, value in dag.items():
    #         for x in value[2]:
    #             dataframes[df_counter] = (key, x)
    #             df_counter +=1

    #         if value[0] == 'resolve':
    #             op_name = value[1][1]
    #             init_params = value[1][5]

    #             ops[key] = {
    #                 'op' : value[1][1] if value[0] == 'resolve' else 'unkown',
    #                 'parameters' : 
    #             }
    #     print(dataframes)
    #     print(ops)


# # Viz stuff
# G = nx.DiGraph(new_dict)

# # Finding roots
# roots = []
# for component in nx.weakly_connected_components(G):
#     G_sub = G.subgraph(component)
#     roots.extend([n for n,d in G_sub.in_degree() if d==0])
# print(roots)

# pos = nx.nx_pydot.graphviz_layout(G, 'dot')

# # Drawing Graph
# # nx.draw(pos, with_labels=True, font_weight='bold')
# nx.draw_networkx(G, pos)
# plt.show()
