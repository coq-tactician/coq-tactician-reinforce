from dataclasses import dataclass, field
import os
import graphviz
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Set, Tuple, Sequence
from pytact.data_reader import Dataset, Definition, Node, ProofState, Original, Discharged, Substituted
import html
import inflection
import capnp
import pytact.graph_api_capnp as graph_api_capnp
import pytact.graph_api_capnp_cython as apic
import networkx as nx
from collections import deque


class TreeNode:
    def __init__(self, value=0, children=None, incoming_edge=None):
        self.value: str = value
        self.children: List[Tuple[str , TreeNode]] = children if children is not None else []

def graph_to_tree(root):
    """
    Converts a graph into a tree-like structure rooted at `root` using the TreeNode class.
    
    Returns:
        TreeNode: The root of the tree.
    """

    def build_tree(node_value):
        if node_value.label.is_rel:
            return  TreeNode(value=node_value.label.which.name)
            
        if edge_value is not None:
            node = TreeNode(value=node_value.label.which.name)
        else:
            node = TreeNode(value=node_value.label.which.name)

        for edge_value , child_value in node_value.children:
            child_node = build_tree(child_value)
            if child_node:
                node.children.append(child_node)

        return node

    return build_tree(root)

def get_candidates(larger_tree , root_node_value):
    """
    Takes a tree and a subtree root node's value and searches for all the candidates in the tree and 
    returns the value
    """
    candidates_lis = []
    q = deque([larger_tree])
    while q:
        l = len(q)
        for i in range(l):
            cur_node = q.pop()
            if cur_node.value == root_node_value:
                candidates_lis.append(cur_node)
            
            for c in cur_node.children:
                q.append(c)
    
    return candidates_lis

def is_subtree(larger_tree: Node , smaller_tree: TreeNode) -> bool:
    def are_identical(tree1: Node , tree2: TreeNode) -> bool:
        if tree2.value == "EVAR":
            return True
        if tree1.label.which.name != tree2.value:
            return False
        if tree1 and tree2.children == []:
            return True

        set_main = {}
        for e1 , c1 in (tree1.children):
            set_main[e1.name] = c1
        
        token_node_children_length = len(tree2.children)
        tree_node_children_length = len(tree1.children)
        #assert token_node_children_length == tree_node_children_length , f"Check the graph at {tree1.label.which.name} and at token {tree2.value}"
        
        c = 0
        for e2 , c2 in (tree2.children):
            if (e2) not in set_main:
                return False    
            if not are_identical(set_main[e2] , c2):
                return False
        return True
    
    # Check if the trees are identical from this node
    if are_identical(larger_tree, smaller_tree):
        return True

    return False

class UrlMaker(ABC):

    @abstractmethod
    def definition(self, fname: Path, defid: int) -> str:
        pass
    @abstractmethod
    def proof(self, fname: Path, defid: int) -> str:
        pass
    @abstractmethod
    def outcome(self, fname: Path, defid: int, stepi: int, outcomei: int) -> str:
        pass
    @abstractmethod
    def global_context(self, fname: Path) -> str:
        pass
    @abstractmethod
    def folder(self, path: Path) -> str:
        pass
    @abstractmethod
    def root_folder(self) -> str:
        pass

@dataclass
class Settings:
    no_defaults: bool = False # Should we use default settings?
    ignore_edges: List[int] = field(default_factory=lambda: [])
    unshare_nodes: List[int] = field(default_factory=lambda: [])
    show_trivial_evar_substs: bool = False
    hide_proof_terms: bool = False
    show_edge_labels: bool = False
    show_tokenization: bool = False
    order_edges: bool = False
    concentrate_edges: bool = False
    show_non_anonymized_tactics: bool = False
    max_depth: int = 0
    max_size: int = 100

    def __post_init__(self):
        if not self.no_defaults:
            self.ignore_edges = [graph_api_capnp.EdgeClassification.schema.enumerants['constOpaqueDef']]
            label = graph_api_capnp.Graph.Node.Label
            self.unshare_nodes = [label.definition.value, label.sortSProp.value,
                                  label.sortProp.value, label.sortSet.value, label.sortType.value]

@dataclass
class GraphVisualizationData:
    data: Dict[Path, Dataset]
    trans_deps: Dict[Path, Set[Path]] = field(init=False)
    graphid2path: List[Path] = field(init=False)

    def __post_init__(self):
        self.trans_deps = transitive_closure({d.filename: list(d.dependencies)
                                              for d in self.data.values()})
        self.graphid2path = [d.filename for d in sorted(self.data.values(), key=lambda d: d.graph)]

@dataclass
class GraphVisualizationOutput:
    svg: str
    location: List[Tuple[str, str]] # tuple[Name, URL]
    active_location: int
    text: List[str] = field(default_factory=lambda: [])
    popups: List[Tuple[str, str]] = field(default_factory=lambda: []) # DOM id, text

def node_label_map(node: Node) -> Tuple[str, str, str]:
    enum = apic.Graph_Node_Label_Which
    label = node.label
    if d := node.definition:
        name = d.name
        return (
            'box', name.split('.')[-1],
            f"{inflection.camelize(node.label.definition.which.name.lower())} {d.name}"
        )
    which = label.which
    if which == enum.SORT_PROP:
        return 'ellipse', 'Prop', 'SortProp'
    elif which == enum.SORT_S_PROP:
        return 'ellipse', 'SProp', 'SortSProp'
    elif which == enum.SORT_SET:
        return 'ellipse', 'Set', 'SortSet'
    elif which == enum.SORT_TYPE:
        return 'ellipse', 'Type', 'SortType'
    elif which == enum.REL:
        return 'circle', '↑', 'rel'
    elif which == enum.PROD:
        return 'circle', '∀', 'prod'
    elif which == enum.LAMBDA:
        return 'circle', 'λ', 'lambda'
    elif which == enum.LET_IN:
        return 'ellipse', 'let', 'LetIn'
    elif which == enum.APP:
        return 'circle', '@', 'app'
    elif which == enum.CASE_BRANCH:
        return 'ellipse', 'branch', 'CaseBranch'
    else:
        name = inflection.camelize(label.which.name.lower())
        return 'ellipse', name, name
    #add evar , evarSubst , cast , caseBranch , fix , 
    # fixFun , coFix , coFixFun
    # int , float , primitive

def truncate_string(data, maximum):
    return data[:(maximum-2)] + '..' if len(data) > maximum else data

def make_label(context, name):
    name_split = name.split('.')
    common = os.path.commonprefix([name_split, context.split('.')])
    return '.'.join(name_split[len(common):])

def graphviz_escape(s):
    return s.replace('\\', '\\\\')

def make_tooltip(d):
    return f"{inflection.camelize(d.node.label.definition.which.name.lower())} {d.name}"

def render_proof_state_text(ps: ProofState):
    return ('<br>'.join(ps.context_text) +
            '<br>----------------------<br>' + ps.conclusion_text +
            '<br><br>Raw: ' + ps.text)

class GraphVisualizator:
    def __init__(self, data: GraphVisualizationData, url_maker: UrlMaker, settings: Settings = Settings()):
        self.data = data.data
        self.trans_deps = data.trans_deps
        self.graphid2path = data.graphid2path
        self.url_maker = url_maker
        self.settings = settings
        self.node_counter = 0


        arrow_heads = [ "dot", "inv", "odot", "invdot", "invodot" ]
        edge_arrow_map = {}
        for group in graph_api_capnp.groupedEdges:
            for i, sort in enumerate(group.conflatable):
                edge_arrow_map[sort.raw] = arrow_heads[i]
        self.edge_arrow_map = edge_arrow_map

    def url_for_path(self, r: Path):
        if r in self.data:
            return r.with_suffix('').parts[-1], self.url_maker.global_context(r)
        elif len(r.parts) > 0:
            return r.parts[-1], self.url_maker.folder(r)
        else:
            return 'dataset', self.url_maker.root_folder()

    def path2location(self, path: Path):
        return [self.url_for_path(parent) for parent in list(reversed(path.parents)) + [path]]

    def dot_apply_style(self, dot):
        dot.attr('node', fontsize='11', fontname="Helvetica", margin='0', height='0.3',
                 style="rounded, filled", penwidth="0")
        dot.attr('edge', fontsize='11', fontname="Helvetica", arrowsize='0.5', penwidth="0.6")
        dot.attr('graph', fontsize='11', fontname="Helvetica", penwidth="0.6", ranksep='0.3')
        if self.settings.order_edges:
            dot.attr('graph', ordering='out')
        if self.settings.concentrate_edges:
            dot.attr('graph', concentrate='true')


    def render_node(self, dot, node: Node, shape: str, label: str, id: Union[str, None] = None,
                    tooltip: Union[str, None] = None):
        if not id:
            id = str(node)
        if not tooltip:
            tooltip = label
        url = None
        if node.definition:
            url = self.url_maker.definition(self.graphid2path[node.graph], node.nodeid)
        dot.node(id, label, URL = url, shape = shape, tooltip = tooltip)
        return id

    def render_file_node(self, dot, f: Path):
        label = f"File: {f.with_suffix('')}"
        node_id = f"file-{f}"
        dot.node(node_id, label, URL = self.url_maker.global_context(f), shape='box')
        return node_id

    def global_context(self, fname: Path):
        dot = graphviz.Digraph(format='svg')
        self.dot_apply_style(dot)
        dot.attr('graph', compound="true")

        dataset = self.data[fname]
        representative = dataset.representative
        module_name = dataset.module_name

        def render_def(dot2, d: Definition):
            label = make_label(module_name, d.name)
            if representative and representative.node == d.node:
                label = "Representative: " + label
            tooltip = make_tooltip(d)
            if isinstance(d.status, Original):
                id = self.render_node(dot2, d.node, 'box', label, tooltip=tooltip)
            elif isinstance(d.status, Discharged):
                id = self.render_node(dot2, d.node, 'box', label, tooltip=tooltip)
                target = d.status.original
                dot.edge(id, repr(target.node),
                         arrowtail="inv", dir="both", constraint="false", style="dashed")
            elif isinstance(d.status, Substituted):
                target = d.status.original
                if d.node.graph == target.node.graph:
                    id = self.render_node(dot2, d.node, 'box', label, tooltip=tooltip)
                    dot.edge(id, str(target.node),
                             arrowtail="odot", dir="both", constraint="false", style="dashed")
                else:
                    with dot2.subgraph() as dot3:
                        dot3.attr(rank='same')
                        id = self.render_node(dot3, d.node, 'box', label, tooltip=tooltip)
                        id2 = self.render_node(dot3, target.node, 'box',
                                               make_label(module_name, target.name),
                                               tooltip=make_tooltip(target))
                        dot.edge(id, id2,
                                 arrowtail="odot", dir="both", constraint="false", style="dashed")

        for cluster in dataset.clustered_definitions():

            start = str(cluster[0].node)
            ltail = None
            if len(cluster) == 1:
                render_def(dot, cluster[0])
            else:
                ltail = "cluster_"+start
                with dot.subgraph(name=ltail) as dot2:
                    dot2.attr('graph', style='rounded')
                    last = None
                    for d in cluster:
                        render_def(dot2, d)
                        if last:
                            dot2.edge(str(last.node), str(d.node), style="invis")
                        last = d

            if prev := cluster[-1].previous:
                target = str(prev.node)
                lhead = None
                if len(list(prev.cluster)) > 1:
                    lhead = "cluster_"+target
                dot.edge(str(cluster[-1].node), target, lhead = lhead, ltail = ltail)
            for fi in cluster[-1].external_previous:
                fid = self.render_file_node(dot, self.graphid2path[fi.node.graph])
                dot.edge(str(cluster[-1].node), fid, ltail = ltail)

        location = self.path2location(fname)
        return GraphVisualizationOutput(dot.source, location, len(location) - 1)


    def add_paranthesis(self, st):
        return "( " + st + ")"
    
    def print_tree(self, node: TreeNode, inside_left_application: bool = False , for_all: bool = False) -> str:
        cur_node_label = node.value
        if cur_node_label == "REL":
            return "↑"
        elif cur_node_label == "EVAR":
            return "."
        elif cur_node_label == "SORT_TYPE":
            return "Type"
        elif cur_node_label == "SORT_PROP":
            return "Prop"
        elif cur_node_label == "CASE_BRANCH":
            return "branch"
        elif cur_node_label == "SORT_SET":
            return "Set"
        elif cur_node_label == "SORT_S_PROP":
            return "SProp"
        elif cur_node_label == "LET_IN":
            return "let"
        elif cur_node_label == "FLOAT":
            return str(node.label.float.value)
        elif cur_node_label == "PRIMITIVE":
            return "PRIMITIVE"
        elif cur_node_label == "INT":
            return str(node.label.int.value)
        elif cur_node_label == "APP":
            application_string = f"{self.print_tree(node.children[0][1] , inside_left_application = True)} {self.print_tree(node.children[1][1])}"
            if not inside_left_application:
                return self.add_paranthesis(application_string)
            return application_string
        elif cur_node_label == "PROD":
            prod_string = f"∀  {self.print_tree(node.children[0][1])} {self.print_tree(node.children[1][1] , for_all=True)}"
            if for_all:
                return prod_string
            return self.add_paranthesis(f"{prod_string}")
        elif cur_node_label == "LAMBDA":
            return self.add_paranthesis(f"λ {self.print_tree(node.children[0][1])}, {self.print_tree(node.children[1][1])}")
        elif cur_node_label == "DEFINITION":
            return cur_node_label
        else:
            child_label_list = f"{cur_node_label}("
        
        lis = []
        for edge , child in node.children:
            child_printed = self.print_tree(child)
            lis.append(child_printed)
        
        child_label_list += ",".join(lis)
            
        child_label_list += ")"
        return child_label_list

    def print_token(self, node: Node , left_child: str ="" , right_child : str = ""):
        cur_node_label = node.label.which.name
        if node.label.is_rel:
            return "↑"
        elif node.label.is_sort_prop:
            return "Prop"
        elif node.label.is_sort_s_prop:
            return "SProp"
        elif node.label.is_sort_type:
            return "Type"
        elif node.label.is_sort_set:
            return "Set"
        elif node.label.is_case_branch:
            cur_node_label = "branch"
        elif node.label.is_let_in:
            cur_node_label = 'let'
        elif node.label.is_app:
            application_string = ""
            if left_child:
                application_string += left_child
            if right_child:
                application_string += " " + right_child
            return self.add_paranthesis(application_string)
        elif node.label.is_prod:
            prod_string = ''
            if left_child:
                prod_string += left_child
            if right_child:
                prod_string += ", "  + right_child if left_child else right_child
            return self.add_paranthesis(f"∀ {prod_string}")
        
        elif node.label.is_lambda_:
            return self.add_paranthesis(f"λ {left_child}, {right_child}") if right_child else  self.add_paranthesis(f"λ {left_child}")
        
        elif node.label.is_definition:
            return node.definition.name.split(".")[-1] + left_child + " " + right_child #remove the left and right child here
        else:
            return cur_node_label
    
    def merge_token(self, node: Node , token: TreeNode) -> List[Tuple[str , Node]]:
        """
        This function takes the matched root node and the token and merges each node part of the token into a single node 
        with children being a union all the individual node's children.
        
        A list called children is returned which contains all the children of the merged token. This list is then used during 
        rendering for visualizing the graph with the merged token. 
        """
        q = deque([token]) # queue for tokem
        node_q = deque([node]) # queue for the graph
        children = []
        while q:
            l = len(q)
            for _ in range(l):
                if len(node_q) == 0:
                    return children
                cur_node = q.popleft()
                cur_node_q = node_q.popleft()
                if cur_node_q.label.is_rel:
                    continue  
                node_set = {}
                for node_edge , node_child in cur_node_q.children:
                    node_set[(node_edge.name , node_child.label.which.name)] = (node_edge , node_child)
                for e , c in cur_node.children:
                    if (e , c.value) in node_set:
                        node_q.append(node_set[(e , c.value)][1])
                        del node_set[(e , c.value)]
                    q.append(c)
                
                for i , j in node_set.items():
                    children.append(j)
                                
        return children

    
    def match(self, node: Node , sample_token_list: List[TreeNode] = None) -> Tuple[List[Tuple[str , Node]] , str]:
        """
        This function takes the current node and a list of tokens and  checks if the token is present starting from the current node. 
        If the token is present , 'print_tree' is called to create a merged representation of the token which is set as the 
        label for the merged token node. 
        """
        found = False
        new_token = ''
        children = []
        for token in sample_token_list:
            if node.label.which.name == token.value:
                found = is_subtree(node , token)
            if found:
                children = self.merge_token(node , token)
                new_token = self.print_tree(token)
                break
            else:
                children = node.children
        
        return children  , new_token

    def get_node_prefix(self , node, enum, prefix, context_prefix):
        which = node.label.which
        if which == enum.proofState:
            node_prefix = context_prefix
        elif which == enum.contextAssum:
            node_prefix = context_prefix
        elif which == enum.contextDef:
            node_prefix = context_prefix
        elif which == enum.evarSubst:
            node_prefix = prefix + context_prefix
        else:
            node_prefix = prefix
        return node_prefix
    
        
    def visualize_term(self, dot, start: Node, depth, depth_ignore: Set[Node] = set(),
                       max_nodes=100, seen: Union[Dict[str, str], None]=None,
                       node_label_map=node_label_map,
                       prefix='', before_prefix='', proof_state_prefix: Dict[int, str] = {}
                       ) -> str:
        
        nodes_left = max_nodes
        seen = {}
        token1 = TreeNode(value='PROD' , 
                          children=[
                                    ("PROD_TYPE" , TreeNode(value='SORT_TYPE')) , 
                                    ("PROD_TERM" , TreeNode(value='PROD' , children=[("PROD_TERM" , TreeNode(value='EVAR')),
                                                                                     ("PROD_TYPE" , TreeNode(value="REL"))
                                                                                                             ]
                                                                                                             ))])
                                                                                                              
        token2 = TreeNode(value='APP' , children=[("APP_FUN" , TreeNode(value='APP', children=[
            ("APP_ARG" , TreeNode(value="REL")),
            ("APP_FUN" , TreeNode(value="REL"))
        ])), 
                                                  ("APP_ARG" , TreeNode(value="REL"))
                                                  
                                                  ])

        token3 = TreeNode(value='PROD' , children=[("PROD_TERM" , TreeNode(value='SORT_S_PROP')), 
                                                  ("PROD_TYPE" , TreeNode(value="REL")) 
                                                  ])

        token4 = TreeNode(value='APP' , children=[("APP_ARG" , TreeNode(value='REL')), 
                                                  ("APP_FUN" , TreeNode(value="REL")) 
                                                  ])

        token5 = TreeNode(value='CASE', children=[("CASE_TERM" , TreeNode(value='REL'))])
        token6 = TreeNode(value="LAMBDA" , children=[
            ("LAMBDA_TYPE" , TreeNode(value="REL")),
            ("LAMBDA_TERM" , TreeNode(value="LAMBDA" , children=[
                ("LAMBDA_TYPE" , TreeNode(value="APP", children=[("APP_ARG" , TreeNode(value='REL')) , ("APP_FUN" , TreeNode(value='EVAR'))])),
                ("LAMBDA_TERM" , TreeNode(value="CASE", children=[("CASE_TERM" , TreeNode(value='REL'))]))
            ])),
        ])

        token7 = TreeNode(value="LET_IN" , children=[
            ("LET_IN_TYPE" , TreeNode(value='DEFINITION')), ("LET_IN_DEF" , TreeNode(value="APP" , children=[("APP_FUN" , TreeNode(value="REL")) , ("APP_ARG" , TreeNode(value="REL"))]))])

        sample_tokens_list = [token1 , token2 , token3 , token4 , token5 , token6 , token7] #tokens from the tokenizer will come here. 
        
        def recurse(node: Node, depth, context_prefix):
            nonlocal seen
            nonlocal nodes_left
            if self.settings.show_tokenization:
                children , new_token = self.match(node,sample_tokens_list)
            else:
                children = node.children
                new_token = ""
                
            if node.label.is_rel:
                children = []
    
            enum = graph_api_capnp.Graph.Node.Label
            node_prefix = self.get_node_prefix(node, enum, prefix, context_prefix)
            id = node_prefix + str(node) 
            if id in seen:
                return seen[id]
            
            dot_id = f"c{self.node_counter}-{id}"
            nodes_left -= 1
            self.node_counter += 1
            if nodes_left < 0:
                id = 'trunc' + str(self.node_counter)
                dot.node(id, 'truncated')
                return id , ""
            
            shape, label, tooltip = node_label_map(node)
            self.render_node(dot, node, shape, label, id=dot_id, tooltip=tooltip)
            if node.definition and not node in depth_ignore:
                depth -= 1
            
            cur_child_label_lis = []
            if depth >= 0:
                if node.label.which == graph_api_capnp.Graph.Node.Label.evar:
                    # Find the evar-id
                    evarid = [c.label.proof_state.value for _, c in node.children
                              if c.label.which == graph_api_capnp.Graph.Node.Label.proofState][0]
                    context_prefix = proof_state_prefix.get(evarid, context_prefix)

                #print("children" , children[0][1].label.which.name)
                for c in children:
                    edge = c[0]
                    child = c[1]
                    if edge in self.settings.ignore_edges:
                        continue
                    if child.label.which == graph_api_capnp.Graph.Node.Label.evarSubst:
                        substs = [s for _, s in child.children]
                        if not self.settings.show_trivial_evar_substs and substs[0] == substs[1]:
                            continue
                    cid, cur_child_label = recurse(child, depth,
                                    before_prefix if edge == graph_api_capnp.EdgeClassification.evarSubstTerm
                                    else context_prefix)
                    
                    cur_child_label_lis.append(cur_child_label)
                    edge_name = inflection.camelize(apic.EdgeClassification(edge).name.lower())
                    if self.settings.show_edge_labels:
                        edge_label = edge_name
                    else:
                        edge_label = ""
                    dot.edge(dot_id, cid, label=edge_label, tooltip=edge_name, labeltooltip=edge_name,
                                arrowtail=self.edge_arrow_map[edge], dir="both")
                    
            # if len(cur_child_label_lis) >= 2:
            #     updated_node_label = self.print_node(node, left_child = cur_child_label_lis[0] , right_child = cur_child_label_lis[1])
            # elif len(cur_child_label_lis) == 1:
            #     updated_node_label = self.print_node(node, left_child = cur_child_label_lis[0])
            # else:
            #     updated_node_label = self.print_node(node)
            shape="rectangle"

            if new_token!="":
                label = new_token

            self.render_node(dot , node , shape , label , id=dot_id , tooltip=tooltip)
            
            seen[id] = (dot_id , new_token)
            if node.label.which_raw in self.settings.unshare_nodes:
                del seen[id]
            return dot_id , new_token
        
        id , final_label = recurse(start, depth, before_prefix)
        
        return id

    def definition(self, fname: Path, definition: int):
        dot = graphviz.Digraph(format='svg')
        self.dot_apply_style(dot)

        start = self.data[fname].node_by_id(definition)
        depth_ignore = set()
        if d := start.definition:
            depth_ignore = {d.node for d in start.definition.cluster}
        depth = self.settings.max_depth
        max_nodes = self.settings.max_size

        self.visualize_term(dot, start, depth=depth, depth_ignore=depth_ignore,
                            max_nodes=max_nodes)

        location = self.path2location(fname)
        ext_location = location
        label = "[not a definition]"
        text = []
        proof = []
        if d := start.definition:
            label = d.name
            text = [f"Type: {d.type_text}"]
            if term := d.term_text:
                text.append(f"Term: {term}")
            if d.proof:
                proof = [("Proof", self.url_maker.proof(fname, definition))]
        ext_location = (
            location +
            [(make_label(self.data[fname].module_name, label),
              self.url_maker.definition(fname, definition))] +
            proof)
        return GraphVisualizationOutput(dot.source, ext_location, len(location), text)

    def proof(self, fname: Path, definition: int):
        node = self.data[fname].node_by_id(definition)
        d = node.definition
        if not d:
            assert False
        proof = d.proof
        if not proof:
            assert False

        dot = graphviz.Digraph(format='svg')
        self.dot_apply_style(dot)
        dot.attr('node', style="filled", fillcolor="white", penwidth="0.6")
        dot.attr('graph', ordering="out")
        surrogates = set()
        outcome_to_id = {}
        for i, step in enumerate(proof):
            for j, outcome in enumerate(step.outcomes):
                id = str(outcome.before.id)
                while id in surrogates:
                    id = id + '-s'
                surrogates.add(id)
                outcome_to_id[(i, j)] = id
        for i, step in enumerate(proof):
            with dot.subgraph(name='cluster_' + str(i)) as dot2:
                dot2.attr('graph', labelloc="b", style="rounded")
                if tactic := step.tactic:
                    if self.settings.show_non_anonymized_tactics:
                        tactic_text = tactic.text_non_anonymous
                    else:
                        tactic_text = tactic.text
                else:
                    tactic_text = 'unknown'
                dot2.attr(label=tactic_text)
                for j, outcome in enumerate(step.outcomes):
                    before_id = outcome_to_id[(i, j)]
                    dot2.node(before_id, label='⬤', shape='circle', fontsize="7", height="0.25pt",
                              URL = self.url_maker.outcome(fname, definition, i, j))
                    for after in outcome.after:
                        if outcome.before.id == after.id:
                            after_id = before_id + '-s'
                            style = 'dashed'
                        else:
                            after_id = str(after.id)
                            style = 'solid'
                        dot.edge(before_id, after_id, style=style)
                    if not outcome.after:
                        qedid = str('qed-'+str(i)+'-'+str(j))
                        dot2.node(qedid, label='', shape='point', height='0.05', fillcolor='black')
                        dot.edge(before_id, qedid)

        location = (self.path2location(fname) +
                    [(make_label(self.data[fname].module_name, d.name),
                      self.url_maker.definition(fname, definition)),
                     ("Proof", self.url_maker.proof(fname, definition))])
        return GraphVisualizationOutput(dot.source, location, len(location) - 1)

    def outcome(self, fname: Path, definition: int, stepi: int, outcomei: int):
        depth = self.settings.max_depth
        max_nodes = self.settings.max_size
        node = self.data[fname].node_by_id(definition)
        d = node.definition
        if not d:
            assert False
        proof = d.proof
        if not proof:
            assert False

        dot = graphviz.Digraph(format='svg')
        self.dot_apply_style(dot)

        outcome = proof[stepi].outcomes[outcomei]
        seen = {}

        def node_label_map_with_ctx_names(context: Sequence[Node],
                                          context_text: Sequence[str]):
            mapping = {n: s for n, s in zip(context, context_text)}
            def nlm(node: Node):
                enum = graph_api_capnp.Graph.Node.Label
                which = node.label.which
                if which == enum.contextAssum:
                    name = graphviz_escape(mapping[node])
                    return 'ellipse', truncate_string(name, 20), f"ContextAssum {name}"
                elif which == enum.contextDef:
                    name = graphviz_escape(mapping[node])
                    return 'ellipse', truncate_string(name, 20), f"ContextDef {name}"
                else:
                        return node_label_map(node)
            return nlm

        popups = []

        with dot.subgraph(name='cluster_before') as dot2:
            ps = outcome.before
            dot2.attr('graph',
                      label=f"Before state\n{graphviz_escape(truncate_string(ps.conclusion_text, 70))}",
                      tooltip=f"Before state {graphviz_escape(ps.conclusion_text)}",
                      id='before-state')
            popups.append(('before-state', render_proof_state_text(ps)))
            prefix = 'before'
            self.visualize_term(dot2, ps.root, depth=depth, prefix=prefix, before_prefix=prefix,
                                max_nodes=max_nodes, seen=seen,
                                node_label_map=node_label_map_with_ctx_names(ps.context, ps.context_text))

        with dot.subgraph(name='cluster_tactic') as dot2:
            prefix = 'before'
            tactic_text = 'unknown'
            tactic_base_text = 'unknown'
            if t := outcome.tactic:
                tactic_text = t.text
                tactic_base_text = (t.base_text.replace('__argument_marker__', '_')
                                    .replace('private_constant_placeholder', '_'))
            dot2.attr('graph', label=f"Tactic\n{tactic_text}")
            dot2.node('tactic', label = tactic_base_text)
            for i, arg in enumerate(outcome.tactic_arguments):
                if arg is None:
                    dot2.node(f"tactic-arg{i}", label=f"arg {i}: unknown")
                else:
                    id = self.visualize_term(dot2, arg, depth=depth, prefix=prefix, before_prefix=prefix,
                                        max_nodes=max_nodes, seen=seen)
                    dot2.node(f"tactic-arg{i}", label=f"arg {i}")
                    dot2.edge(f"tactic-arg{i}", id)
                dot2.edge('tactic', f"tactic-arg{i}")


        for ai, after in enumerate(outcome.after):
            with dot.subgraph(name='cluster_after' + str(ai)) as dot2:
                dot2.attr('graph',
                          label=f"After state {ai}\n{graphviz_escape(truncate_string(after.conclusion_text, 70))}",
                          tooltip=f"After state {ai} {graphviz_escape(after.conclusion_text)}",
                          id=f'after-state{ai}')
                popups.append((f'after-state{ai}', render_proof_state_text(after)))
                prefix = f'after{ai}'
                self.visualize_term(dot2, after.root, depth=depth, prefix=prefix, before_prefix=prefix,
                                    max_nodes=max_nodes, seen=seen,
                                    node_label_map=node_label_map_with_ctx_names(after.context, after.context_text))

        if not self.settings.hide_proof_terms:
            with dot.subgraph(name='cluster_term') as dot2:
                dot2.attr('graph',
                          label=f"Proof term\n{graphviz_escape(truncate_string(outcome.term_text, 70))}",
                          tooltip=f"Proof term {graphviz_escape(outcome.term_text)}",
                          id='proof-term')
                popups.append(('proof-term', outcome.term_text))
                prefix = 'term'
                proof_state_prefix = {after.id: f'after{ai}' for ai, after in enumerate(outcome.after)}
                id = self.visualize_term(dot2, outcome.term, depth=depth, prefix=prefix, before_prefix='before',
                                    proof_state_prefix=proof_state_prefix,
                                    max_nodes=max_nodes, seen=seen)
                # Sometimes the subgraph is completely empty because the term is contained in another subgraph.
                # Therefore, we artificially add a extra root node
                dot2.node('artificial-root', 'TermRoot')
                dot2.edge('artificial-root', id)

        location = (self.path2location(fname) +
                    [(make_label(self.data[fname].module_name, d.name),
                      self.url_maker.definition(fname, definition)),
                     ("Proof", self.url_maker.proof(fname, definition)),
                     (f"Step {stepi} outcome {outcomei}",
                      self.url_maker.outcome(fname, definition, stepi, outcomei))])
        return GraphVisualizationOutput(dot.source, location, len(location) - 1, [], popups)

    def folder(self, expand_path: Path) -> GraphVisualizationOutput:
        expand_parts = expand_path.parts

        dot = graphviz.Digraph(engine='dot', format='svg')
        self.dot_apply_style(dot)

        hierarchy = {'files': [], 'subdirs': {}}
        for f in self.data:
            dirs = f.parent.parts
            leaf = hierarchy
            for d in dirs:
                leaf['subdirs'].setdefault(d, {'files': [], 'subdirs': {}})
                leaf = leaf['subdirs'][d]
            leaf['files'].append(f)
        def common_length(p1, p2):
            return len(os.path.commonprefix([p1, p2]))
        def retrieve_edges(rel, h, depth: int):
            for n in h['files']:
                deps = {Path(*d.parts[:depth+1]) for d in self.trans_deps[n]
                        if common_length(d.parts, expand_parts) == depth}
                rel[Path(*n.parts[:depth+1])] |= deps
            for d in h['subdirs']:
                retrieve_edges(rel, h['subdirs'][d], depth)
            return rel
        def tunnel_hierarchy(dot, h, depth):
            if depth == len(expand_parts):
                rel = retrieve_edges(defaultdict(set), h, depth)
                (rel, repr) = transitive_reduction(rel)
                for n in rel:
                    if len(repr[n]) == 1:
                        label, url = self.url_for_path(n)
                        shape = 'box'
                    else:
                        label = '<<table border="0" cellborder="0" cellpadding="7">'
                        for r in repr[n]:
                            slabel, surl = self.url_for_path(r)
                            label += f'<tr><td href="{html.escape(surl)}">{slabel}</td></tr>'
                        label += "</table>>"
                        url = None
                        shape = 'plaintext'
                    dot.node(str(n), label, URL = url, shape = shape)
                for n, deps in rel.items():
                    for d in deps:
                        dot.edge(str(n), str(d))
            else:
                for d in h['subdirs']:
                    cur_path: Path = Path(*expand_parts[:depth]) / d
                    if depth < len(expand_parts) and d == expand_parts[depth]:
                        cluster_name = 'cluster_' + str(cur_path)
                        with dot.subgraph(name=cluster_name) as dot2:
                            dot2.attr('graph', style="filled", fillcolor="white", label=d,
                                      URL = self.url_maker.folder(Path(*expand_parts[:depth+1])))
                            tunnel_hierarchy(dot2, h['subdirs'][d], depth+1)

        with dot.subgraph(name='cluster_root') as dot2:
            dot2.attr('graph', style="filled", fillcolor="white", label='dataset',
                      URL = self.url_maker.root_folder())
            tunnel_hierarchy(dot2, hierarchy, 0)

        location = self.path2location(expand_path)
        return GraphVisualizationOutput(dot.source, location, len(location) - 1)

def transitive_closure(rel):
    trans_deps = defaultdict(set)
    def calc_trans_deps(a, n):
        for d in rel[n]:
            if d not in trans_deps[a]:
                trans_deps[a].add(d)
                calc_trans_deps(a, d)
    for n in list(rel.keys()):
        calc_trans_deps(n, n)
    return trans_deps

def transitive_reduction(rel):
    """This is not a real transitive reduction (NP-hard when it needs to be a subgraph).
    This is an approximation where all strongly connected components are smashed together
    in one node.
    """
    trans_deps = transitive_closure(rel)
    repr_items = defaultdict(set)
    representative = {}
    def calc_components(n):
        if n in representative:
            return
        repr_items[n].add(n)
        representative[n] = n
        for d in trans_deps[n]:
            if n in trans_deps[d]:
                repr_items[n].add(d)
                representative[d] = n
    for n in list(rel.keys()):
        calc_components(n)
    repr_trans_rel = defaultdict(set)
    def calc_new_trans_deps(n):
        for d in trans_deps[n]:
            if n not in trans_deps[d]:
                repr_trans_rel[n].add(representative[d])
    for n in list(rel.keys()):
        calc_new_trans_deps(n)
    def calc_sparse_deps(n):
        res = set()
        for d in repr_trans_rel[n]:
            k = [d2 for d2 in repr_trans_rel[n]
                 if d != d2 and d in trans_deps[d2]]
            if not k:
                res.add(d)
        return res
    k = {n: calc_sparse_deps(n) for n in repr_items}
    return (k, repr_items)
