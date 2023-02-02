#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Set

import torch
from torch import Graph
from torch.fx import GraphModule, Node, symbolic_trace

from kernl.utils.fx import static_args_are_equal


# Originaly taken from
# https://github.com/pytorch/pytorch/blob/a27a4a02fecfdd626b25794a84954731b80f29fb/torch/fx/passes/utils/matcher_utils.py


@dataclass
class InternalMatch:
    # Nodes from which the match was found
    anchors: List[Node]
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node] = field(default_factory=dict)

    module_map: Dict[str, str] = field(default_factory=dict)

    # nodes in target graph that are matched placeholder in pattern
    placeholder_nodes: List[Node] = field(default_factory=list)

    # nodes in matched subgraph returned by output
    returning_nodes: List[Node] = field(default_factory=list)

    def __copy__(self):
        return InternalMatch(
            anchors=self.anchors,
            nodes_map=self.nodes_map.copy(),
            placeholder_nodes=self.placeholder_nodes.copy(),
            returning_nodes=self.returning_nodes.copy(),
        )


class SubgraphMatcher:
    def __init__(
        self,
        pattern: Graph,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
    ) -> None:
        """
        Args:
            pattern: the targeted matching pattern, represented in fx.Graph.
            match_output: If True, output node in the pattern graph will be treated as a part of the targeted pattern.
                If False, output node is ignored during match.
            match_placeholder: If True, placeholder node in the pattern graph will be treated as a part of
                the targeted pattern. If False, placeholder nodes will be used a wildcard.
            remove_overlapping_matches: If True, in the case of overlapping matches, only the first match
                will be returned.
        """

        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder
        self.remove_overlapping_matches = remove_overlapping_matches

        if len(pattern.nodes) == 0:
            raise ValueError("SubgraphMatcher cannot be initialized with an empty pattern")

        for node in pattern.nodes:
            if node.op != "output":
                assert len(node.users) > 0, "SubgraphMatcher cannot be initialized with an pattern with dead code"

        # TODO: assert pattern is a connected graph

        self.pattern_placeholder_nodes = [n for n in pattern.nodes if n.op == "placeholder"]
        output_node = next(iter(reversed(pattern.nodes)))
        # nodes returned by outputs
        self.pattern_returning_nodes: List[Node] = output_node.all_input_nodes

        self.pattern_anchors: List[Node] = []
        if match_output:
            self.pattern_anchors = [output_node]
        else:
            # If a node has output_node as the ONLY user, then this node is a graph sink,
            # and should be matched against as an anchor
            self.pattern_anchors = [n for n in output_node.all_input_nodes if len(n.users) == 1]

    def _nodes_are_equal(self, pn: Node, gn: Node) -> bool:
        # TODO: match args and kwargs

        # if exact match for placeholder is not required, then use placeholder as a wildcard
        if not self.match_placeholder and pn.op == "placeholder":
            return True

        if pn.op == gn.op:
            if pn.op == "placeholder" or pn.op == "output":
                return True
            # CHANGE HERE
            # We check arguments
            if not static_args_are_equal(pn, gn):
                return False
            # If it's call_module we only compare the type
            # We will replace the module from the replacement pattern with the old one
            if pn.op == "call_module":
                pn_mod = dict(pn.graph.owning_module.named_modules())
                gn_mod = dict(gn.graph.owning_module.named_modules())
                pn_mod_type = type(pn_mod[pn.target])
                gn_mod_type = type(gn_mod[gn.target])
                if pn_mod_type == gn_mod_type:
                    return True

            return pn.target == gn.target
        return False

    def _is_contained(self, nodes_map: Dict[Node, Node]) -> bool:
        # `lookup` represents all the nodes in `original_graph`
        # that are part of `pattern`
        lookup: Dict[Node, Node] = {gn: pn for pn, gn in nodes_map.items()}
        for gn, pn in lookup.items():
            # Placeholders can be used by other nodes in the graphs
            if pn.op == "placeholder":
                continue

            # nodes returned by output are allowed to be used in other areas of the graph
            if pn in self.pattern_returning_nodes:
                continue

            for user in gn.users:
                # If this node has users that were not in `lookup`, then it must leak out of the
                # pattern subgraph
                if user not in lookup:
                    return False
        return True

    def _remove_overlapping_matches(self, matches: List[InternalMatch]) -> List[InternalMatch]:
        non_overlapping_matches: List[InternalMatch] = list()
        nodes_matched: Set[Node] = set()

        for match in matches:
            found_overlap = False
            for pn, gn in match.nodes_map.items():
                if pn.op not in {"placeholder", "output"} and gn in nodes_matched:
                    found_overlap = True
                    break

            if not found_overlap:
                non_overlapping_matches.append(match)
                for pn, gn in match.nodes_map.items():
                    if pn.op not in {"placeholder", "output"}:
                        nodes_matched.add(gn)
        return non_overlapping_matches

    def _match_nodes(self, pn: Node, gn: Node, match: InternalMatch) -> bool:
        # Check if we've already matched these nodes in the current
        # traversal
        if pn in match.nodes_map:
            return match.nodes_map[pn] == gn

        # TODO: use a more efficienty way to check if gn is matched before: two-way dict
        if gn in match.nodes_map.values():
            return False

        if not self._nodes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        match.nodes_map[pn] = gn

        if pn.op == "call_module":
            match.module_map[pn.target] = gn.target

        if pn.op == "placeholder":
            return True

        # Recursively traverse upwards to check if `pn` is a true
        # match for `gn`
        match_found = len(pn.all_input_nodes) == len(gn.all_input_nodes) and all(
            self._match_nodes(pn_, gn_, match) for pn_, gn_ in zip(pn.all_input_nodes, gn.all_input_nodes)
        )

        if not match_found:
            match.nodes_map.pop(pn)
            return False

        return True

    def match(self, graph: Graph) -> List[InternalMatch]:
        """
        Returns:
            The matched subgraphs.
                Thre returned subgraph would be fully self-contained, meaning the nodes (except placeholder
                and nodes returned by output) can only be consumed by nodes within the matched subgraph.

        Subgraph pattern matcher is implemented with the backtracking style in the following steps:

        1. We first identify all the anchor nodes in the pattern graph. The anchor nodes
        are the "sinks" (nodes with no user other than the output node) of the pattern graph.
        One pattern graph could have multiple anchors if it has multiple return values.

        2. In the target graph, we identify the potential candidate nodes that can be matched
        with each anchor. These anchor-candidate pairs are the starting points for
        pairwise per-node matching.

        3. For each anchor-candidate pair, we simultaneously traverse backwards (DFS) in both
        pattern and target graphs. For every pattern nodes along traversal path, we compare it
        against the target nodes. In case any comparison failed, the match for this anchor-candidate
        pair fails. A match is found when DFS completes traversing the graph. See `self._match_nodes`
        for more details.

        4. In the case of multiple anchors, every anchor will need to find a match using step 3.
        In addition, the matches found between anchors need to have a common intersection node
        in order for the match to be valid. This is implemented with backtracking. See `backtracking`
        for more details.

        Note:
            graph traversal must be done in the reverser order because a tensor can have multiple
            consumers, but can only have a single producer. Only with reverser order, we can we jointly
            traverse the pattern and target graph in a deterministic path.
            Warning: In theory, this backtracking algorithm have an **exponential** time complexity. However,
            in practice, it's unlikely to blow up.
        """

        # find candidate nodes to match with pattern anchors
        match_candidates: Dict[Node, List[Node]] = defaultdict(list)
        for pattern_anchor in self.pattern_anchors:
            for node in graph.nodes:
                if self._nodes_are_equal(pattern_anchor, node):
                    match_candidates[pattern_anchor].append(node)
        match_candidates_list = list(match_candidates.items())
        matches: List[InternalMatch] = []
        if len(match_candidates_list) == 0:
            return []

        def backtracking(anchor_index, match):
            if anchor_index == len(match_candidates_list):
                match.placeholder_nodes = [match.nodes_map[pn] for pn in self.pattern_placeholder_nodes]
                match.returning_nodes = [match.nodes_map[pn] for pn in self.pattern_returning_nodes]
                matches.append(match)
                return

            pattern_anchor, candidate_nodes = match_candidates_list[anchor_index]
            saved_match = copy.copy(match)

            for node in candidate_nodes:
                match_found = self._match_nodes(pattern_anchor, node, match)
                if match_found:
                    # match next anchor
                    backtracking(anchor_index + 1, match)

                    # revert to saved_match before matching with current anchor
                match = copy.copy(saved_match)

        match = InternalMatch(anchors=self.pattern_anchors)
        backtracking(0, match)

        # filter out the matches where the subgraph is not fully_contained
        matches = [match for match in matches if self._is_contained(match.nodes_map)]

        if self.remove_overlapping_matches:
            matches = self._remove_overlapping_matches(matches)

        return matches


class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]


def _replace_submodules(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_submodule(mod: torch.nn.Module, target: str) -> Optional[torch.nn.Module]:
        try:
            mod_match = mod.get_submodule(target)
            return mod_match
        except AttributeError:
            return None

    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":
            # CHANGE HERE
            module_name = node.target
            if node.op == "get_attr":
                if "." in module_name:
                    mod, attr = module_name.split(".")
                    module_name = mod
                else:
                    module_name = ""

            gm_submod = try_get_submodule(gm, module_name)

            replacement_submod = try_get_submodule(replacement, module_name)

            # CASE 1: This target already exists as a submodule in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_submod is not None:
                continue

            # CASE 2: The target exists as a submodule in `replacement`
            # only, so we need to copy it over.
            elif replacement_submod is not None:
                new_submod = copy.deepcopy(getattr(replacement, module_name))
                gm.add_submodule(module_name, new_submod)

            # CASE 3: The target doesn't exist as a submodule in `gm`
            # or `replacement`
            else:
                raise RuntimeError(
                    'Attempted to create a "',
                    node.op,
                    '" node during subgraph rewriting '
                    f"with target {node.target}, but "
                    "the referenced submodule does not "
                    "exist in either the original "
                    "GraphModule `gm` or the replacement"
                    " GraphModule `replacement`",
                )

    gm.graph.lint()


def replace_pattern(gm: GraphModule, pattern: Callable, replacement: Callable) -> List[Match]:
    """Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        gm: The GraphModule that wraps the Graph to operate on
        pattern: The subgraph to match in ``gm`` for replacement
        replacement: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
            in the original graph that ``pattern`` was matched to. The list
            is empty if there are no matches. ``Match`` is defined as:
            ``` { .py }
            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]
            ```

    Examples:
    ``` { .py }
    import torch
    from torch.fx import symbolic_trace, subgraph_rewriter
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, w1, w2):
            m1 = torch.cat([w1, w2]).sum()
            m2 = torch.cat([w1, w2]).sum()
            return x + torch.max(m1) + torch.max(m2)
    def pattern(w1, w2):
        return torch.cat([w1, w2]).sum()
    def replacement(w1, w2):
        return torch.stack([w1, w2])
    traced_module = symbolic_trace(M())
    subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)
    ```
    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).
    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    is the parameter that appears directly after ``self``, while the
    last Node is whatever the function returns.)

    One important thing to note is that the parameters of the
    ``pattern`` Callable must be used in the Callable itself,
    and the parameters of the ``replacement`` Callable must match
    the pattern. The first rule is why, in the above code block, the
    ``forward`` function has parameters ``x, w1, w2``, but the
    ``pattern`` function only has parameters ``w1, w2``. ``pattern``
    doesn't use ``x``, so it shouldn't specify ``x`` as a parameter.
    As an example of the second rule, consider replacing
    ``` { .py }
    def pattern(x, y):
        return torch.neg(x) + torch.relu(y)
    ```
    with
    ``` { .py }
    def replacement(x, y):
        return torch.relu(x)
    ```
    In this case, ``replacement`` needs the same number of parameters
    as ``pattern`` (both ``x`` and ``y``), even though the parameter
    ``y`` isn't used in ``replacement``.
    After calling ``subgraph_rewriter.replace_pattern``, the generated
    Python code looks like this:
    ``` { .py }
    def forward(self, x, w1, w2):
        stack_1 = torch.stack([w1, w2])
        sum_1 = stack_1.sum()
        stack_2 = torch.stack([w1, w2])
        sum_2 = stack_2.sum()
        max_1 = torch.max(sum_1)
        add_1 = x + max_1
        max_2 = torch.max(sum_2)
        add_2 = add_1 + max_2
        return add_2
    ```
    """

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph
    pattern_graph: Graph = symbolic_trace(pattern).graph

    matcher = SubgraphMatcher(
        pattern_graph, match_output=False, match_placeholder=False, remove_overlapping_matches=True
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = {}

    replacement_graph_module: GraphModule = symbolic_trace(replacement)
    for match in _matches:
        replacement_graph = copy.deepcopy(replacement_graph_module).graph
        replacement_placeholders = [n for n in replacement_graph.nodes if n.op == "placeholder"]

        # CHANGE HERE
        # We ensure we use the original modules
        for node in replacement_graph.nodes:
            if node.op == "call_module" or node.op == "get_attr":
                submodule_name = node.target
                if node.op == "get_attr":
                    submodule_name = node.target.split(".")[0]
                if submodule_name in match.module_map:
                    to_replace = match.module_map[submodule_name]
                    node.target = node.target.replace(submodule_name, to_replace, 1)
        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            val_map[rn] = match_changed_node.get(gn, gn)

        # Copy the replacement graph over
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            for user in n.users:
                user_nodes.add(user)
        assert user_nodes, "The returning_nodes should have at least one user node"

        if len(user_nodes) == 1:
            first_user_node = list(user_nodes)[0]
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        with original_graph.inserting_before(first_user_node):
            copied_returning_nodes = original_graph.graph_copy(replacement_graph, val_map)

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes,)

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        assert len(match.returning_nodes) == len(copied_returning_nodes)
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_submodules(gm, replacement)

    # Convert _matches: InternalMatch to Match to comply with backward compatibility of this function
    matches: List[Match] = [Match(anchor=match.anchors[0], nodes_map=match.nodes_map) for match in _matches]
    return matches
