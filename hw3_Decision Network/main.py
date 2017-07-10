import logging
from copy import deepcopy
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s - %(lineno)s -%(funcName)s')


import itertools
try:  # math.isclose was added in Python 3.5; but we might be in 3.4
    from math import isclose
except ImportError:
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        """Return true if numbers a and b are close to each other."""
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, varname='?', freqs=None):
        """If freqs is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized."""
        self.prob = {}
        self.varname = varname
        self.values = []
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        """Given a value, return P(value)."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        """Set P(val) = p."""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('{}: ' + numfmt).format(v, p)
                          for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({} {})".format(self.varname, self.prob)


class BayesNode:
    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        """X is a variable name, and parents a sequence of variable
        names or a space-separated string.  cpt, the conditional
        probability table, takes one of these forms:
        * A number, the unconditional probability P(X=true). You can
          use this form when there are no parents.
        * A dict {v: p, ...}, the conditional probability distribution
          P(X=true | parent=v) = p. When there's just one parent.
        * A dict {(v1, v2, ...): p, ...}, the distribution P(X=true |
          parent1=v1, parent2=v2, ...) = p. Each key must have as many
          values as there are parents. You can use this form always;
          the first two are just conveniences.
        In all cases the probability of X being false is left implicit,
        since it follows from P(X=true).
        >>> X = BayesNode('X', '', 0.2)
        >>> Y = BayesNode('Y', 'P', {T: 0.2, F: 0.7})
        >>> Z = BayesNode('Z', 'P Q',
        ...    {(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
        """
        if isinstance(parents, str):
            parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): cpt}
            logging.debug(cpt)
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}
                logging.debug(cpt)

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1
        logging.debug(parents)

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""

        # logging.debug(value)
        assert isinstance(value, bool)
        # logging.debug(self.cpt)
        ptrue = self.cpt[event_values(event, self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event):
        """Sample from the distribution for this variable conditioned
        on event's values for parent_variables. That is, return True/False
        at random according with the conditional probability given the
        parents."""
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents), self.cpt))


def event_values(event, variables):
    """Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])



class BayesNet:
    """Bayesian network containing only boolean-variable nodes."""

    def __init__(self, node_specs=[]):
        """Nodes must be ordered with parents before children."""
        self.nodes = []
        self.decision_nodes = []
        self.variables = []
        self.utility_parents = []
        self.utility_assignment = dict()
        for node_spec in node_specs:
            if node_spec[0] == 'chance':
                self.add(node_spec[1:])
            elif node_spec[0] == 'decision':
                self.decision_nodes.append(node_spec[1])
                self.add(node_spec[1:])
                # self.add(node_spec[1:])
            elif node_spec[0] == 'utility':
                logging.debug(node_spec)
                for parent in node_spec[2].split():
                    self.utility_parents.append(parent)
                self.utility_assignment = node_spec[3]
                # logging.debug(self.utility_parents)
                # logging.debug(self.utility_assignment)


    def add(self, node_spec):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        node = BayesNode(*node_spec)
        # logging.debug(node)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        """Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var):
        """Return the domain of var."""
        if isinstance(var, str):
            return [True, False]
        else:
            return itertools.product([True, False], repeat=len(var))


    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)

def extend(s, var, val):
    """Copy the substitution s and extend it by setting var to val; return copy."""
    # logging.debug(var)
    # logging.debug(val)
    # logging.debug(s)
    s2 = s.copy()
    if isinstance(var, str):
        s2[var] = val
    else:
        for i, v in enumerate(var):
            s2[v] = val[i] 
    # logging.debug(s2)
    return s2


class Factor:
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwise_product(self, other, bn):
        """Multiply two factors, combining their variables."""
        variables = list(set(self.variables) | set(other.variables))
        cpt = {event_values(e, variables): self.p(e) * other.p(e)
               for e in all_events(variables, bn, {})}
        # logging.debug(variables)
        # logging.debug(cpt)
        return Factor(variables, cpt)

    def sum_out(self, var, bn):
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {event_values(e, variables): sum(self.p(extend(e, var, val))
                                               for val in bn.variable_values(var))
               for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def normalize(self):
        """Return probabilities distribution for variables."""
        logging.debug({k:v for (k, v) in self.cpt.items()})
        logging.debug(self.variables)
        return ProbDist(self.variables, {k: v for (k, v) in self.cpt.items()}).normalize()

    def p(self, e):
        """Look up my value tabulated for e."""
        return self.cpt[event_values(e, self.variables)]

def all_events(variables, bn, e):
    """Yield every way of extending e with values for all variables."""
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in all_events(rest, bn, e):
            for x in bn.variable_values(X):
                yield extend(e1, X, x)

def is_hidden(var, X, e):
    """Is var a hidden variable when querying P(X|e)?"""
    # return var != X and var not in e
    logging.debug(var)
    logging.debug('not in')
    logging.debug(X)
    logging.debug(e)
    logging.debug(var not in X and var not in e)
    return var not in X and var not in e

def pointwise_product(factors, bn):
    from functools import reduce
    # logging.debug(factors)
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)


def make_factor(var, e, bn):
    """Return the factor for var in bn's joint distribution given e.
    That is, bn's full joint distribution, projected to accord with e,
    is the pointwise product of these factors for bn's variables."""
    logging.debug(var)
    node = bn.variable_node(var)
    variables = [X for X in [var] + node.parents if X not in e]
    logging.debug(variables)
    logging.debug(e)
    logging.debug({event_values(e1, variables): e1[var] for e1 in all_events(variables, bn, e)})
    cpt = {event_values(e1, variables): node.p(e1[var], e1)
           for e1 in all_events(variables, bn, e)}
    logging.debug(cpt)
    return Factor(variables, cpt)

def sum_out(var, factors, bn):
    """Eliminate var from all factors by summing over its values."""
    result, var_factors = [], []
    for f in factors:
        (var_factors if var in f.variables else result).append(f)
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result


def elimination_ask(X, e, bn):
    """Compute bn's P(X|e) by variable elimination. [Figure 14.11]
    >>> elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    factors = []
    # logging.debug(e)
    # logging.debug(X)
    # logging.debug(bn)
    logging.debug(bn.variables)
    for var in reversed(bn.variables):
        if var not in bn.decision_nodes:
            factors.append(make_factor(var, e, bn))
            if is_hidden(var, X, e):
                logging.debug('is hidden')
                logging.debug(var)
                factors = sum_out(var, factors, bn)
    probability = pointwise_product(factors, bn).normalize()
    logging.debug(probability.prob)
    logging.debug(probability.varname)
    logging.debug(probability.values)
    return probability
 


def enumerate_all(variables, e, bn):
    """Return the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than variables). Parents must precede children in variables."""
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]

    logging.debug(bn.decision_nodes)
    logging.debug(bn.variables)
    logging.debug(Y)
    if Y in bn.decision_nodes:
        # logging.debug('not in chance nodes')
        return enumerate_all(rest, e, bn)
    Ynode = bn.variable_node(Y)
    # logging.debug(Ynode)
    # logging.debug(e)
    # logging.debug(rest)
    # logging.debug([y for y in bn.variable_values(Y)])
    # logging.debug(Ynode.p(y, e))

    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        # logging.debug(Ynode)
        # logging.debug(Y)
        # logging.debug([y for y in bn.variable_values(Y)])

        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))

def enumeration_ask(X, e, bn):
    """Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn. [Figure 14.9]
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    # logging.debug(X)
    Q = ProbDist(X)

    # logging.debug(e)
    # logging.debug(X)
    # logging.debug(bn.variables)
    for xi in bn.variable_values(X):
        # logging.debug(xi)
	Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    # logging.debug(Q.normalize())

    return Q.normalize()

import re
def parse_assign(line):
    # logging.debug(line)
    variable = line.split(',')
    assignment = {}
    variables = []
    for var in variable:
        match = re.search(r'(\w*)\s=\s([\+,\-])', var)
        if match:
            assignment[match.group(1)] = True if match.group(2) == '+' else False
            variables.append(match.group(1))
        else:
            # logging.debug(var)
            variables.append(var.strip())
    variables = tuple(variables)
    # logging.debug(variables)
    return variables, assignment

class QuerySpec:
    def __init__(self, querytype, X, e):
        self.querytype = querytype
        self.X = X
        self.e = e


def parse_variable_evidence(line):
    line = line[line.find('(') + 1:-1]
    parts = line.split('|')
    
    variables, assignment = parse_assign(parts[0])
    X = (variables, assignment)
    logging.debug(X)
    e = None
    if len(parts) == 2:
        e = parse_assign(parts[1])[1]
        logging.debug(e)
    return X, e

def parse_query(line):
    """P(X|e)
       X is variables
       e is evidence
       P(G = +)
       P(G = - | B = +, A = +)
       P(D = +, F = +, E = +)
       EU(A = +, C = + | B = +)
       MEU(A, C)
       MEU(A, C | B = -)

       decision
       A

       utility | E F G
       100 + + +
       50 + + -
       50 + - +
       50 - + +
       0 + - -
       0 - + -
       0 - - +
       -100 - - -
    """
    qs = None
    # logging.debug(line)
    
    X, e = parse_variable_evidence(line)
    logging.debug(X)
    logging.debug(e)
    if line.startswith('P'):
        qs = QuerySpec('P', X, e)

    elif line.startswith('EU'):
        qs = QuerySpec('EU', X, e)

    elif line.startswith('MEU'):
        qs = QuerySpec('MEU', X, e)
    else:
        assert 'not match'
    return qs


def parse_bayes_node(lines):
    """
    >>> X = BayesNode('X', '', 0.2)
    >>> Y = BayesNode('Y', 'P', {T: 0.2, F: 0.7})
    >>> Z = BayesNode('Z', 'P Q',
    ...    {(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
    
    """
    logging.debug(lines)

    parents = '' 
    if lines[1] == 'decision':
        logging.debug(lines[0])
        return ['decision', lines[0], '', dict()]
    
    parts = lines[0].split('|')
    variable = re.search(r'\w*', parts[0]).group()

    if len(parts) == 2:
        parents = parts[1][1:]

    cpt = dict()
    for line in lines[1:]:
        match = re.search(r'([\-\w\.]+)\s*([\+\-\s]*)', line)
        if match:
            combination = [val == '+' for val in match.group(2).split(' ')]
            val = float(match.group(1))
            combination = tuple(combination)
                # logging.debug(combination)
            if '+' not in match.group(2) and '-' not in match.group(2):
                cpt = float(match.group(1)) 
            else:
                cpt[combination] = val
    logging.debug((variable, type(variable)))
    logging.debug((parents, type(parents)))
    logging.debug(cpt)

    if variable == 'utility':
        # logging.debug(cpt)
        return ['utility', variable, parents, cpt]
    else:
        # logging.debug(cpt)
        return ['chance', variable, parents, cpt]

def pr(query, bn):
    X = query.X
    e = query.e
    logging.debug(X)
    logging.debug(e)
    # logging.debug(bn.variables)
    if e is None:
        return enumerate_all(bn.variables, X[1], bn)
    else:
        # logging.debug(X[0])
        # logging.debug(X[1])
        logging.debug(e)
        varnames = []
        logging.debug(bn.variables)
        conditional_prob_dist = elimination_ask(X[0], e, bn)
        # conditional_prob_dist = enumeration_ask(X[0], e, bn)
        varnames = conditional_prob_dist.varname
        logging.debug(varnames)
        logging.debug(e)
        logging.debug(X[1])
        logging.debug(bn)
        hidden_decision_nodes = []
        for ele in varnames:
            if ele not in X[1]:
                hidden_decision_nodes.append(ele)
        factors = []
        logging.debug(hidden_decision_nodes)
        
        vals = list()
        for assignment in itertools.product([True, False], repeat=len(hidden_decision_nodes)):
            val = list()
            i = 0
            for ele in varnames:
                if ele in X[1]:
                    val.append(X[1][ele])
                else:
                    val.append(assignment[i])
                    i += 1
            vals.append(tuple(val))
        return sum(conditional_prob_dist[val] for val in vals)

def get_assignment(unassign_parents, evidence, bn):
    # logging.debug(unassign_parents)
    # logging.debug(evidence)
    parent_assignment_dicts, parent_tuples = [], []
    for unassign_parents_assignment in itertools.product([True, False], repeat=len(unassign_parents)):
        unassign_parents_assign_dict = {}
        for i, unassign_parent in enumerate(unassign_parents):
            unassign_parents_assign_dict[unassign_parent] = unassign_parents_assignment[i]
        parents_truth_tuple = []
        for v in bn.utility_parents:
            if v in unassign_parents:
                parents_truth_tuple.append(unassign_parents_assign_dict[v])
            if v in evidence:
                parents_truth_tuple.append(evidence[v])
        parent_assignment_dicts.append(unassign_parents_assign_dict)
        parent_tuples.append(tuple(parents_truth_tuple))
        
    return parent_assignment_dicts, parent_tuples



def eu(query, bn):
    evidence = {}
    X = query.X
    e = query.e
    # logging.debug(X)
    # logging.debug(e)
    evidence.update(query.X[1])
    if e is not None:
        evidence.update(query.e)
    # utility_parents = bn.utility_parents
    # logging.debug(utility_parents)
    unassign_parents = [p for p in bn.utility_parents if p not in evidence]
    # logging.debug(unassign_parents)
    # logging.debug(evidence)
    parent_assignment_dicts, parent_tuples =  get_assignment(unassign_parents, evidence, bn)
    # logging.debug(parent_assignment_dicts)
    # logging.debug(parent_tuples)
    unassign_parents = tuple(unassign_parents)
    total_utility = 0.0
    for assignment, utility_truth_table in zip(parent_assignment_dicts, parent_tuples):
        X = (unassign_parents, assignment)
        logging.debug(X)
        # logging.debug(bn)
        # logging.debug(evidence)
        q = QuerySpec('P', X, evidence)
        # logging.debug(utility_truth_table)
        # logging.debug(bn.utility_assignment)
        # logging.debug(pr(q, bn))     
        logging.debug(bn.utility_assignment[utility_truth_table])
        total_utility += pr(q, bn) * bn.utility_assignment[utility_truth_table]
    return total_utility


def meu(query, bn):
    evidence ={}
    X = query.X
    # logging.debug(X[0])
    # logging.debug(len(X[0]))
    # logging.debug(type(X[0]))
    e = query.e
    evidence.update(query.X[1])
    if e is not None:
        evidence.update(e)
    decisions = [node for node in bn.decision_nodes if node not in evidence]
    max_utility = float('-inf')
    truth_vals = ''
    for assignment in itertools.product([True, False], repeat=len(decisions)):
        new_evidence = evidence.copy()
        for i, decision in enumerate(decisions):
            new_evidence[decision] = assignment[i]
        q = QuerySpec('EU', ('', []), new_evidence)
        eu_value = eu(q, bn)
        # logging.debug(X[0])
        # logging.debug(new_evidence)
        # logging.debug(eu_value)
        if eu_value > max_utility:
            max_utility = eu_value
            temp_truth_val = ''
            for var in X[0]:
                # logging.debug(var)
                # logging.debug(new_evidence)
                if new_evidence[var]:
                    temp_truth_val += '+ '
                else:
                    temp_truth_val += '- '
            truth_vals = temp_truth_val
    return '{}{}'.format(truth_vals, int(round(max_utility)))

    

if __name__ == '__main__':
    file_num = '03'
    # input_file = open('testcases/input'+ file_num+'.txt', 'r')
    input_file = open('input'+ file_num+'.txt', 'r')


    lines = []
    node_start = False
    queries = []
    bayes_specs= []
    for line in input_file.readlines():
        line = line.strip()
        if not node_start and line != "******":
            q = parse_query(line)
            queries.append(q)
        if node_start == False and line == "******":
            node_start = True
            continue

        if line == "***" or line == "******":
            specs = parse_bayes_node(lines)
            bayes_specs.append(specs)

            lines = []
            continue

        if node_start:
            lines.append(line)
    bayes_specs.append(parse_bayes_node(lines))
    bn = BayesNet(bayes_specs)
    f = open('output.txt', 'w')

    
    for q in queries[:]:
        if q.querytype == 'P':
            logging.debug(q.X)
            logging.debug(q.e)
            # from decimal import Decimal
            # print Decimal(pr(q, bn))
            result = '{:.2f}'.format(round(pr(q, bn)+0.0000001, 2))

            logging.debug(result)
            f.write(result)

        if q.querytype == 'EU':
            logging.debug('in Eu')
            logging.debug(q.X)
            logging.debug(q.e)
            result = round(eu(q, bn))
            result = str(int(result))
            f.write(result)

        if q.querytype == 'MEU':
            result = meu(q, bn)
            f.write(result)
        f.write('\n')
    f.close()
    print '--------my output file--------'

    input_file = open('output.txt', 'r')
    for line in input_file.readlines():
        line = line.strip()
        print line

    
    print '------------------'
    input_file = open('output'+ file_num+'.txt', 'r')
    for line in input_file.readlines():
        line = line.strip()
        print line


