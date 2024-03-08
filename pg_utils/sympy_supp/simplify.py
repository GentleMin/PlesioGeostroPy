# -*- coding: utf-8 -*-
"""
Additional simplification functions
"""

import sympy
from typing import List, Union, Callable, Optional


def recursive_collect(expr: sympy.Expr, 
    syms: List[Union[sympy.Symbol, List[sympy.Symbol]]], 
    evaluate: bool = True) -> Union[dict, sympy.Expr]:
    """Recursively collect an expression by symbols
    
    :param sympy.Expr expr: expression to be processed
    :param List[Union[sympy.Symbol, List[sympy.Symbol]]] syms: sympy
        symbols, whose coefficients are to be collected
    :param bool evaluate: if True, the output will be
        an expression, grouped by collected terms;
        if False, the output will be a dict representing
        a tree with symbol-coeff key-value pairs. Output
        {
            sym1: term1,
            sym2: term2,
            ...
            symN: termN
        }
        represents an expression
        sym1*term1 + sym2*term2 + ... symN*termN
        Each `term` can also be a dict, leading to the
        tree structure
    
    :returns: collected output
    """
    collected_tree = recursive_collect_tree(expr, syms)
    if evaluate:
        return recursive_eval_tree(collected_tree)
    else:
        return collected_tree


def recursive_collect_tree(expr: sympy.Expr, 
    syms: List[Union[sympy.Symbol, List[sympy.Symbol]]]) -> dict:
    """Recursively collect an expression by symbols to a tree
    
    :param sympy.Expr expr: expression to be processed
    :param List[Union[sympy.Symbol, List[sympy.Symbol]]] syms: sympy
        symbols whose coefficients are to be collected.
        See :py:func:`recursive_collect` for details.
    
    :returns: collected output as a tree repr. by a dict
    """
    recursive_depth = len(syms)
    # If the final layer, then return sympy.collect result as dict
    if recursive_depth == 1:
        return sympy.collect(expr, syms=syms[0], evaluate=False)
    # If not the final layer, then collect by leading symbol
    collected_terms = sympy.collect(expr, syms=syms[0], evaluate=False)
    # Recursively collect the coefficient terms
    next_syms = syms[1:]
    collected_terms = {
        sym: recursive_collect_tree(term, syms=next_syms)
        for sym, term in collected_terms.items()
    }
    # Combine to form the output
    return collected_terms


def recursive_eval_tree(collected_tree: dict) -> sympy.Expr:
    """Recursively evaluate a tree
    
    :param dict collected_tree: collected tree dictionary
        See :py:func:`recursive_collect` for details.
    
    :returns: collected expression
    """
    # If the tree branch is an expression, it is a leaf node
    if isinstance(collected_tree, sympy.Expr):
        return collected_tree
    # Else the tree branch is not leaf; divide and conquer
    elif isinstance(collected_tree, dict):
        add_args = [
            sympy.Mul(sym, recursive_eval_tree(term), evaluate=True)
            for sym, term in collected_tree.items()
        ]
        return sympy.Add(*add_args, evaluate=True)
    else:
        raise TypeError


def process_leaf_node(collected_tree: dict, 
    leaf_op: Callable[[sympy.Expr], sympy.Expr]) -> None:
    """Process the leaf node in a collected tree
    
    :param dict collected_tree: collected tree dictionary
    """
    for sym in collected_tree:
        if isinstance(collected_tree[sym], sympy.Expr):
            collected_tree[sym] = leaf_op(collected_tree[sym])
        elif isinstance(collected_tree[sym], dict):
            process_leaf_node(collected_tree[sym], leaf_op)
        else:
            raise TypeError
