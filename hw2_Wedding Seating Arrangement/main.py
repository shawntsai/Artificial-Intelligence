from collections import Counter
import sys
import random
FRIEND = 1
ENEMY = -1


class Literal:
    # assume literal is positive by default
    def __init__(self, guest, table, is_negation=False):
        self.symbol = (guest, table)
        self.is_negation = is_negation
        self.hashcode = -1

    def is_compliment(self, literal):
        if self.symbol == literal.symbol and self.is_negation is not literal.is_negation:
            return True
        else:
            return False
    def __str__(self):
        return str(('~' + str(self.symbol) if self.is_negation else self.symbol))
    

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, self.__class__):
            return False
        return self.symbol == other.symbol and self.is_negation == other.is_negation
    
    def __hash__(self):
        if self.hashcode == -1:
            self.hashcode = 17  
            self.hashcode = self.hashcode * 37 + hash('-' if self.is_negation else '+')
            self.hashcode = self.hashcode * 37 + hash(self.symbol)
        return self.hashcode

    def eval_literal(self, val):
        if self.is_negation:
            return not val 
        else:
            return val 
    
    def is_same_symbol(self, literal):
        if not literal or not literal.symbol:
            return False
        elif self.symbol != literal.symbol:
            return False
        else:
            return True


class Clause:
    def __init__(self, literals=None):
        self.hashcode = -1
        self.cached_positive_symbols= set()
        self.cached_negative_symbols= set()
        self.cached_toutology_result = None
        if literals:
            self.literals = literals
            for literal in literals:
                if not literal.is_negation:
                    self.cached_positive_symbols.add(literal.symbol)
                else:
                    self.cached_negative_symbols.add(literal.symbol)
        else:
            self.literals = list()

    def __hash__(self):
        if self.hashcode == -1:
            self.hashcode = 1
            for l in self.literals:
                self.hashcode  = 31 * self.hashcode + hash(l)
        return self.hashcode 

    def __eq__(self, other):
        if not other:
            return False
        if len(self.literals) != len(other.literals):
            return False
        return Counter(self.literals) == Counter(other.literals)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str([str(x) for x in self.literals])


    def is_empty(self):
        return len(self.literals) == 0
    
    def is_false(self):
        return self.is_empty()

    def is_tautology(self):
        if self.cached_toutology_result is None:
            if len(self.cached_positive_symbols.intersection(self.cached_negative_symbols)) >= 1:
                self.cached_toutology_result = True
                return True
            else:
                self.cached_toutology_result = False
                return False
        else:
            return self.cached_toutology_result
    
    def is_unit_clause(self):
        return len(self.literals) == 1

    def add_literal(self, literal):
        self.literals.append(literal)
        if not literal.is_negation:
            self.cached_positive_symbols.add(literal.symbol)
        else:
            self.cached_negative_symbols.add(literal.symbol)

    def contain_one_compliment_literal(self, clause):
        count = 0 
        for literal in self.literals:
            for to_be_compare in clause.literals:
                if literal.is_compliment(to_be_compare):
                    count += 1
                if count > 1:
                    return False
        if count == 0:
            return False 
        else:
            return True 

    def is_same_literals(self, clause):
        other_literals = clause.literals
        if len(self.literals) != len(other_literals):
            return False

        for literal in self.literals:
            if literal not in other_literals:
                return False
        return True

    # check if atleast one literal satisfied in clause
    def satisfiable(self, random_model):
        for literal in self.literals:
            val = random_model[literal.symbol] 
            if literal.eval_literal(val):
                return True
        return False
        
    def contain_literal(self, literal):
        for l in self.literals:
            if l == literal:
                return True
        return False



def get_one_table_cons(guests, tables):
    result = set()
    # print 'at least'
    # every get sits
    for i in range(1, guests + 1):
        clause = Clause()
        for j in range(1, tables + 1):
            literal = Literal(i, j)
            clause.add_literal(literal)
        result.add(clause)
    
    # at most sitting on one table
    for i in range(1, guests + 1):
        for j in range(1, tables + 1):
            for k in range(j + 1, tables + 1):
                clause = Clause()
                literal_left = Literal(i, j, True)
                clause.add_literal(literal_left)
                literal_right = Literal(i, k, True)
                clause.add_literal(literal_right)
                result.add(clause)
    return result
 
       

def get_freind_cons(friend_x, friend_y, table):
    result = set()
    # print 'get_freind_cons'
    # print friend_x, friend_y
    for i in range(1, table + 1):
        clause_left, clause_right = Clause(), Clause()
        literal_left = Literal(friend_x, i, True)
        literal_right = Literal(friend_y, i, False)
        clause_left.add_literal(literal_left)
        clause_left.add_literal(literal_right)

        # clause_right = Clause()
        literal_left = Literal(friend_y, i, True) 
        literal_right = Literal(friend_x, i, False)
        clause_right.add_literal(literal_left)
        clause_right.add_literal(literal_right)

        result.add(clause_left)
        result.add(clause_right)
    return result



def get_enemy_cons(enemy_x, enemy_y, table):
    result = set()
    for i in range(1, table+ 1):
        clause = Clause()
        literal_left = Literal(enemy_x, i, True)
        clause.add_literal(literal_left)
        literal_right = Literal(enemy_y, i, True)
        clause.add_literal(literal_right)
        result.add(clause)

    return result



def generate_relationship_mat(guests, friends, enemies):
    relationship_mat = [[0] * (guests + 1) for _ in range(guests + 1)]
    for friend_pair in friends:
        x, y = friend_pair
        relationship_mat[x][y] = FRIEND
        relationship_mat[y][x] = FRIEND
    for enemy_pair in enemies:
        x, y = enemy_pair
        relationship_mat[x][y] = ENEMY
        relationship_mat[y][x] = ENEMY
    

    return relationship_mat

def get_friends_enemies_cons(guests, relationship_mat, tables):
    cons_friends = set()
    cons_enimies = set()

    for i in range(1, guests + 1):
        for j in range(i, guests +1):
            if relationship_mat[i][j] == FRIEND:
                cons_friends.update(get_freind_cons(i, j, tables))

            elif relationship_mat[i][j] == ENEMY:
                cons_enimies.update(get_enemy_cons(i, j, tables))
    return cons_friends, cons_enimies



def generate_KB(guests, tables, relationship_mat):
    KB = set()
    cons_one_table = get_one_table_cons(guests, tables)
    cons_friends, cons_enimies = get_friends_enemies_cons(guests, relationship_mat, tables)

    KB.update(cons_one_table)
    KB.update(cons_friends)
    KB.update(cons_enimies)

    return KB



class PLRResolution:
    def __init__(self, kb):
        self.clauses = set(kb)
        self.discard_tautologies()
    
    def discard_tautologies(self):
        to_discard = set()
        for clause in self.clauses:
            if clause.is_tautology():
                to_discard.add(clause)
        # print 'discard', len(to_discard)
        self.clauses = self.clauses.difference(to_discard)


    def is_satisfiable(self):
        times = 0

        while(True):

            new_clauses = set()
            times += 1
            # print 'time'
            # print times
            # print len(self.clauses)
            clauses_as_list = list(self.clauses)

            for i in range(len(clauses_as_list) - 1):
                clause = clauses_as_list[i]
                for j in range(i+1, len(clauses_as_list)):
                    other_clause = clauses_as_list[j]

                    resolvents = self.plResolve(clause, other_clause)

                    if self.contains_empty_clause(resolvents):
                        return False
                    new_clauses.update(resolvents)
            if new_clause.issubset(self.clauses):
                return True
            self.clauses.update(new_clauses)   


    def plResolve(self, ci, cj):
        resolvents = set()
        self.resolve_positive_with_negative(ci, cj, resolvents)
        self.resolve_positive_with_negative(cj, ci, resolvents)
        return resolvents

    def contains_empty_clause(self, clauses):
        for c in clauses:
            if c.is_empty():
                return True
        return False
    
    def resolve_positive_with_negative(self, clause_1, clause_2, resolvents):
        complementary = clause_1.cached_positive_symbols.intersection(clause_2.cached_negative_symbols)
        for complement in complementary:
            resolvent_literals = []

            for c1l in clause_1.literals:
                if c1l.is_negation or not c1l.symbol == complement:
                    resolvent_literals.append(c1l)
            for c2l in clause_2.literals:
                if not c2l.is_negation or not c2l.symbol == complement:
                    resolvent_literals.append(c2l)

            resolvent_clause = Clause(resolvent_literals)

            if not resolvent_clause.is_tautology():
                resolvents.add(resolvent_clause)


    def add_to_map(self, clause, other_clause):
        if clause in self.resolved_map:


            pair_clauses = self.resolved_map[clause]
            pair_clauses.add(other_clause)
            self.resolved_map[clause] = pair_clauses
        else:
            pair_clauses = set()
            pair_clauses.add(other_clause)
            self.resolved_map[clause] = pair_clauses

    def add_resoved(self, clause, other_clause):
        self.add_to_map(clause, other_clause)
        self.add_to_map(other_clause, clause)
        
                            
    def contains_new_clause(self, new_clause, resolved_clauses):
        for clause in resolved_clauses:
            if clause.is_same_literals(new_clause):
                # print 'same literal'
                return True
        for clause in self.clauses:
            if clause.is_same_literals(new_clause):
                # print 'same literal'
                return True
        return False



def generate_random_model(number_guests, number_tables):
    random_model = {}

    for i in range(1, number_guests + 1):
        for j in range(1, number_tables + 1):
            p = random.random()
            if p > 0.5:
                random_model[(i, j)] = True
            else:
                random_model[(i, j)] = False
    return random_model 

def model_satisfied_KB(KB, random_model, unsatisfied_clauses):
    for clause in KB:
        if not clause.satisfiable(random_model):
            unsatisfied_clauses.append(clause)
    # print 'unsatisfied_clauses length', len(unsatisfied_clauses)
    return False if unsatisfied_clauses else True

def walk_sat(KB, max_flips, number_guests, number_tables, probability):
    random_model = generate_random_model(number_guests, number_tables) 
    group = random_model.keys()
    # print group
    group.sort(key=lambda tup: tup[0])
    # print 'in walk sort'
    # print group 

    while (max_flips > 0):
        unsatisfied_clauses = list()
        max_flips -= 1
        if not model_satisfied_KB(KB, random_model, unsatisfied_clauses):
            random_index = random.randint(0, len(unsatisfied_clauses) - 1)
            unsatisfied_clause = unsatisfied_clauses[random_index]
            p = random.random()
            if p > probability:
                random_model = maximize_satisfied_clauses(KB, unsatisfied_clause, random_model)
            else:
                random_model = random_flip(unsatisfied_clause, random_model)
        else:
            return random_model

    return None 

def random_flip(unsatisfied_clause, random_model):
    literals = unsatisfied_clause.literals
    index = random.randint(0, len(literals) - 1)
    literal = literals[index]
    val = random_model[literal.symbol]
    if not literal.eval_literal(val):
        random_model[literal.symbol] = not val
    return random_model


def maximize_satisfied_clauses(KB, unsatisfied_clause, random_model):
    max_satisfied_size = 0
    max_literal = None
    satisfied_clauses_dict = dict(random_model)

    for literal in unsatisfied_clause.literals:
        val = satisfied_clauses_dict[literal.symbol]
        if not literal.eval_literal(val):
            satisfied_clauses_dict[literal.symbol] = not val
     
        satisfied_size = count_satisfied_clause(KB, satisfied_clauses_dict)
        if max_satisfied_size < satisfied_size:
            max_literal = literal
        satisfied_clauses_dict[literal.symbol] = val
   
    val = satisfied_clauses_dict[max_literal.symbol]
    if not literal.eval_literal(val):
        satisfied_clauses_dict[literal.symbol] = not val
    return satisfied_clauses_dict

def count_satisfied_clause(KB, random_model):
    count = 0
    for clause in KB:
        if clause.satisfiable(random_model):
            count += 1
    return count

     
class Model:
    def __init__(self):
    # dict for proposition symbol and truth value
        self.assignments = {}

    def union(self, symbol, b):
        self.assignments[symbol] = b
        return self

    """
    determine based on the current assignments with the model, whether a clause
    is known to be true, false or unknown

    """
    def determine_value(self, c):
        result = None # unknown
        if c.is_tautology():
            result = True
        elif c.is_false():
            result = False
        else:
            unassigned_symbols = False
            for positive in c.cached_positive_symbols:
                value = None
                if positive in self.assignments:
                    value = self.assignments[positive]
                if value != None:
                    if value is True:
                        result = True
                        break
                else:
                    # print 'not assign'
                    unassigned_symbols = True
            if result == None:
                for negative in c.cached_negative_symbols:
                    value = None
                    if negative in self.assignments:
                        value = self.assignments[negative]
                    if value != None:
                        if value is False:
                            result = True
                            break
                    else:
                        # print 'not in assignment'
                        unassigned_symbols = True
                if result == None:
                    if not unassigned_symbols:
                        result = False
        return result
    
    def get_value(self, symbol):
        if symbol in self.assignments:
            return self.assignments[symbol]
        else:
            return None


    def satisfies(self, clauses):
        for c in clauses:
            val = self.determine_value(c) 
            if val is False: 
                return False
        return True



class DPLL:
    def __init__(self, kb):
        self.clauses = kb

    def dpll_satisfiable(self):
        symbols = self.get_proposition_symbols(self.clauses)
        return self.dpll(self.clauses, symbols, Model())

    def get_proposition_symbols(self, clauses):
        result = set()
        for clause in clauses:
            for l in clause.literals:
                result.add(l.symbol)
        return list(result)
   
    def dpll(self, clauses, symbols, model):
        unknown_clauses = []
        # print 'model:'
        for c in clauses:
            val = model.determine_value(c) 
            if val is False:
                return False
            if val is None:
                unknown_clauses.append(c)

        if not unknown_clauses:
            return True

        p, value = self.find_pure_symbol(symbols, unknown_clauses)
        # print p, value

        if p:
            return self.dpll(clauses, self.minus(symbols, p), model.union(p, value))
        p, value = self.find_unit_clause(clauses, model)

        if p:
            return self.dpll(clauses, self.minus(symbols, p), model.union(p, value))
        p, rest = symbols[0], symbols[1:]

        
        return (self.dpll(clauses, rest, model.union(p, True))
                or self.dpll(clauses, rest, model.union(p, False)))


    def find_pure_symbol(self, symbols, clauses):
        for s in symbols:
            found_positive = False
            found_negative = False

            # print '-----pure------------------------'
            for c in clauses:
                # print c.cached_positive_symbols
                # print c.cached_negative_symbols
                # print 'test symbol', s
                # print c
                if not found_positive and s in c.cached_positive_symbols:
                    # print s, 'in positive'
                    found_positive = True
                if not found_negative and s in c.cached_negative_symbols:
                    # print s, 'in_negative'
                    found_negative = True
            # print 'prin 678', found_positive, found_positive
            if found_positive != found_negative:
                # print 'found one pure'
                # print s, found_positive
                return s, found_positive
        return None, None
    
    def find_unit_clause(self, clauses, model):
        # print '-----------unit-----------------'
        # print 'test find unit'
        # for c in clauses:
            # print [(x.symbol, x.is_negation) for x in c.literals]
        # print 'my model'
        # print model.assignments
        def unit_clauses_assign(clause, model):
            p, value = None, None
            for l in clause.literals:
                symbol, positive = l.symbol, not l.is_negation
                if symbol in model.assignments:
                    if model.assignments[symbol] == positive:
                        return None, None
                elif p:
                    return None, None
                else:
                    p, value = symbol, positive
            # print 'assign ', p, value
            return p, value
        for c in clauses:
            p, value = unit_clauses_assign(c, model)
            if p:
                # print 'got one unit'
                # print p, value
                return p, value
        return None, None
                



    def every_clause_true(self, clauses, model):
        return model.satisfies(clauses)

    def some_clause_false(self, clauses, model):
        for c in clauses:
            if model.determine_value(c) is False:
                # print '717', 
                # for l in c.literals:
                    # print l.symbol
                return True
        return False


    def minus(self, symbols, p):
        return [x for x in symbols if x != p]


if __name__ == '__main__':
     
    input_file = open('input2.txt', "r")
    number_guests, number_tables = input_file.readline().split(" ")
    number_guests = int(number_guests)
    number_tables = int(number_tables)
    # print 'number of guest', number_guests
    # print 'number of tables', number_tables
    friends = []
    enimies = []
    lines = input_file.read().splitlines()
    for line in lines:
        # print line
        (x, y, relationship) = line.split(" ")
        # print 'relationship', relationship
        if relationship == "F":
            friends.append((int(x), int(y)))
        if relationship == "E":
            enimies.append((int(x), int(y)))
    relationship_mat = generate_relationship_mat(number_guests, friends, enimies)      
    # import numpy as np
    # print np.matrix(relationship_mat)

    KB = generate_KB(number_guests, number_tables, relationship_mat)
   
    # pl = PLRResolution(KB)
    # pl_satisfiable = pl.is_satisfiable()
    # print pl_satisfiable


    dpll = DPLL(KB)
    dpll_satisfiable = dpll.dpll_satisfiable()

    f = open('output.txt', "w")
    f.write('yes\n' if dpll_satisfiable else 'no\n')
    if dpll_satisfiable:
        model = None
        while(model == None):
            model = walk_sat(KB, sys.maxint, number_guests, number_tables, 0.5)    
        if  model:
            group = {k: v for k, v in model.iteritems() if v == True}    
            # print group 
            pairs = group.keys()
            # sorted(pairs)
            for pair in sorted(pairs):
                f.write(str(pair[0])+ " "+ str(pair[1]) + "\n")

        else:
            print 'max flips not enough'

    f.close()
           
