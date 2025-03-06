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


C_tmp = sympy.Symbol(r'C')


def summands_lindep(expr: sympy.Expr, term_expr: sympy.Expr):
    """Collect the linear coefficient in an expression
    """
    collected = expr.subs({term_expr: C_tmp*term_expr}).doit()
    return collected.coeff(C_tmp)


def summands_dep(expr: sympy.Expr, term_expr: sympy.Expr):
    """Collect all terms in a summation that are dependent on expression 
    """
    collected = expr.as_independent(term_expr, as_Add=True)[1]
    return collected


def collect_by_type(expr: sympy.Expr, base_type, **kwargs):
    """Group all terms in a summation by types
    """
    return sympy.collect(expr, expr.atoms(base_type), **kwargs)


def collect_jacobi(expr: sympy.Expr, **kwargs):
    """Collect all terms by Jacobi polynomials
    """
    return collect_by_type(expr, sympy.jacobi, **kwargs)


from sympy.polys.polyoptions import allowed_flags
from sympy.polys.polytools import poly_from_expr


def horner_delayed(f, *gens, **args):
    """Evaluation-delayed Horner transformation
    """
    allowed_flags(args, [])
    F, _ = poly_from_expr(f, *gens, **args)
    form, gen = sympy.S.Zero, F.gen
    
    if F.is_univariate:
        for coeff in F.all_coeffs():
            form = sympy.Add(form*gen, coeff, evaluate=False)
    else:
        F, gens = sympy.Poly(F, gen), gens[1:]
        for coeff in F.all_coeffs():
            form = sympy.Add(form*gen, horner_delayed(f, *gens, **args), evaluate=False)
    
    return form
    
