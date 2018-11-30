---  
tag: AI 
---

# Introduction

Testing $ KB \models q$ can be done by testing unsatisfiability of $( KB \land \lnot q)$

Three algorithems based model checking to achieve this:
- Truth table enumeration
- Complete backtracking search: DPLL algorithm
- Incomplte local search: WALKSAT algorithm

# Truth table enumeration
- Enumerate all possible models for all symbols
- Models $\leftarrow$ Figure out models where KB is true
- check if q is true in all Models.

# DPLL algorithm
Determin if an input propositional logic sentence in CNF is satisfiable.
Improvements over truth table enumeration:
- early termination    
A sentence is False if Any clause in sentence is False.     
A clause is True if any literal is True.     
- Pure symbol heuristic     
A pure symbol: always appears with same 'sign' in all clauses.
Make a pure symbol literal corresponding value.
- Unit clause heristic     
The only literal in a Unit clause must be the corresponding value.

## Implementation


```python
import random
import sys

class Clause():
    """
      Gridworld
    """
    def __init__(self, literals):
        self.clause = literals
        self.symbols = list(literals.keys())

    def getSymbols(self):
        return self.symbols

    def getClause(self):
        return self.clause

    def print(self):
        print(self.clause)

if __name__ == '__main__':
    # -D v -B v C
    literals = {'D':False, 'B':False, 'C':True}
    clause1= Clause(literals)
```


```python
import random
import sys
from clause import Clause

flatten = lambda l: [item for sublist in l for item in sublist]

def checkClause(clause, model):
    '''
    check if model satisfies the clause

    True: satisfied
    False: unsatisfied
    -1: Unkown
    '''

    # A v -B
    # model{A: true, B: true}
    symbols = clause.getSymbols()
    unknow = False
    for s in symbols:
        value_model = model.get(s, None)
        value_should = clause.getClause()[s]
        if (value_model == value_should):
            return True

        if(value_model == None):
            unknow = True
    if(unknow):
        return -1
    else:
        return False

def checkClauses(clauses, model):
    '''
    check if mode satisfies all the clauses
    True: satisfied
    False: unsatisfied
    -1: Unkown
    '''
    # print('------checkClauses-------')
    ret = True
    for clause in clauses:
        # clause.print()
        flag = checkClause(clause, model)
        # print('---', flag)
        if (flag == False):
            return False # return False if any clause if False
        if (flag == -1):
            ret = False
    if(ret != True):
        return -1
    else:
        return True

def FIND_PURE_SYMBOL(symbols, clauses, model):
    # print('-----FIND_PURE_SYMBOL-----')
    # if it is already in model, discard it
    for symbol in symbols:
        value_model = model.get(symbol, None)
        if(value_model != None):
            continue

        values = [c.getClause().get(symbol, None) for c in clauses]
        # print(values)
        length = len(values)-values.count(None)
        len_true = values.count(True)
        len_false = values.count(False)
        # print(length,len_true,len_false)
        if( length==len_true ):
            return (symbol,True)
        if( length==len_false ):
            return (symbol, False)

    return None, None

def FIND_UNIT_CLAUSE(clauses, model):
    # print('-----FIND_UNIT_CLAUSE-----')
    for c in clauses:
        if (len(c.getSymbols())==1):
            s = c.getSymbols()[0]
            if (s not in list(model.keys())):
                return s,c.getClause()[s]
    return None, None


def DPLL_SATISFIABLE(clauses):
    symbols = [c.getSymbols() for c in clasues]
    symbols = flatten(symbols)
    symbols = set(symbols) # unique
    print('symbols:', symbols)

    return DPLL(clauses, symbols, {})
 
def DPLL(clauses, symbols, model):
    print('DPLL', model)
    # if every clause in clauses is true in the model, then return True
    flag = checkClauses(clauses, model)
    # print('checkClauses:', flag)
    if(flag == True):
        return (True, model)

    # if some clauses is false in model then return false
    for clasue in clauses:
        flag = checkClause(clasue, model)
        if(flag == False):
            return (False, model)

    # find pure symbol
    p, value = FIND_PURE_SYMBOL(symbols, clauses, model)
    if(p != None):
        symbols.remove(p)
        newmodel = model.copy()
        newmodel[p] = value
        return DPLL(clauses, symbols, newmodel)

    # find unit clause. the only literal in a unit clause
    p, value = FIND_UNIT_CLAUSE(clauses, model)
    if(p != None):
        symbols.remove(p)
        newmodel = model.copy()
        newmodel[p] = value
        return DPLL(clauses, symbols, newmodel)

    # the other symbols can be True or False
    if(len(symbols) > 0):
        s = symbols.pop();

        newmodel1 = model.copy()
        newmodel1[s] = True
        ret1, model1 = DPLL(clauses, symbols, newmodel1)

        newmodel2 = model.copy()
        newmodel2[s] = False
        ret2, model2 =DPLL(clauses, symbols, newmodel2)

        if(ret1): 
            return ret1, model1
        if(ret2):
            return ret2, model2

    return False, model


clasues = []

# testcase1
# # A v -B
# literals1 = {'A':True, 'B':False}
# clause1= Clause(literals1)
# clasues.append(clause1)

# # -B v -C
# literals2 = {'B':False, 'C':False}
# clause2= Clause(literals2)
# clasues.append(clause2)

# # C v A
# literals3 = {'C':True, 'A':True}
# clause3= Clause(literals3)
# clasues.append(clause3)

# testcase2
clasues.append(Clause({'D':False, 'B':False, 'C':True}))
clasues.append(Clause({'B':True, 'A':False, 'C':False}))
clasues.append(Clause({'B':False, 'E':True, 'C':False}))
clasues.append(Clause({'B':True, 'E':True, 'D':False}))
clasues.append(Clause({'B':True, 'E':True, 'C':False}))
clasues.append(Clause({'E':False, 'F':True, 'A':False}))
clasues.append(Clause({'E':True, 'F':False, 'A':True}))
clasues.append(Clause({'G':False}))

for c in clasues:
    c.print()

satisfiable, model = DPLL_SATISFIABLE(clasues)
print('Final satisfiable ? ', satisfiable, model)
```

    {'D': False, 'B': False, 'C': True}
    {'B': True, 'A': False, 'C': False}
    {'B': False, 'E': True, 'C': False}
    {'B': True, 'E': True, 'D': False}
    {'B': True, 'E': True, 'C': False}
    {'E': False, 'F': True, 'A': False}
    {'E': True, 'F': False, 'A': True}
    {'G': False}
    symbols: {'F', 'B', 'C', 'A', 'D', 'E', 'G'}
    DPLL {}
    DPLL {'D': False}
    DPLL {'D': False, 'G': False}
    DPLL {'D': False, 'G': False, 'F': True}
    DPLL {'D': False, 'G': False, 'F': True, 'B': True}
    DPLL {'D': False, 'G': False, 'F': True, 'B': True, 'C': True}
    DPLL {'D': False, 'G': False, 'F': True, 'B': True, 'C': True, 'A': True}
    DPLL {'D': False, 'G': False, 'F': True, 'B': True, 'C': True, 'A': True, 'E': True}
    DPLL {'D': False, 'G': False, 'F': True, 'B': True, 'C': True, 'A': True, 'E': False}
    DPLL {'D': False, 'G': False, 'F': True, 'B': True, 'C': True, 'A': False}
    DPLL {'D': False, 'G': False, 'F': True, 'B': True, 'C': False}
    DPLL {'D': False, 'G': False, 'F': True, 'B': False}
    DPLL {'D': False, 'G': False, 'F': False}
    Final satisfiable ?  True {'D': False, 'G': False, 'F': True, 'B': True, 'C': True, 'A': True, 'E': True}


# WALKSAT
on every iteration, the algorithm picks an unsatisfied clause and picks a symbol in the clasue to flip
return a model or failure.    

failure: two possible causes    
- the sentence is unsatisifiable
- the algorithm needs more time to find the model(max_flips)

## Implementation


```python


import random
import sys
from clause import Clause

flatten = lambda l: [item for sublist in l for item in sublist]

def checkClause(clause, model):
    '''
    check if model satisfies the clause
    '''

    # A v -B
    # model{A: true, B: true}
    symbols = clause.getSymbols()
    for s in symbols:
        value_model = model[s]
        value_should = clause.getClause()[s]
        if (value_model == value_should):
            return True
    return False

def checkClauses(clauses, model):
    '''
    check if mode satisfies all the clauses
    '''
    for clause in clauses:
        flag = checkClause(clause, model)
        if (not flag):
            return False # return False if any clause if False
    return True

def WALKSAT(clauses, p, maxz_flip):
    symbols = [c.getSymbols() for c in clasues]
    symbols = flatten(symbols)
    symbols = set(symbols) # unique
    print('symbols:', symbols)
    
    # randomly aissign value to each symbol
    model = {}
    for symbol in symbols:
        model[symbol] = (random.random()>0.5)
    print('inital model:', model)

    for i in range(maxz_flip):
        #check
        flag = checkClauses(clauses, model)
        clause_unsatisfied = []
        if(flag): # satisfied
            return model
        else:
            for clause in clauses:
                flag = checkClause(clause, model)
                if (not flag):
                    clause_unsatisfied.append(clause)

        # flip
        # print('clause_unsatisfied:', len(clause_unsatisfied))

        random_int = random.randint(0,len(clause_unsatisfied)-1)
        chosen_clause = clause_unsatisfied[random_int]

        # simplied version
        # simply filp the chosen symbol
        random_int = random.randint(0,len(chosen_clause.getSymbols())-1)
        chosen_symbol = chosen_clause.getSymbols()[random_int]
        model[chosen_symbol] = (not model[chosen_symbol])
        print(i, ' iteration:', model)

    return None # Failure

clasues = []

# testcase1
# # A v -B
# literals1 = {'A':True, 'B':False}
# clause1= Clause(literals1)
# clasues.append(clause1)

# # -B v -C
# literals2 = {'B':False, 'C':False}
# clause2= Clause(literals2)
# clasues.append(clause2)

# # C v A
# literals3 = {'C':True, 'A':True}
# clause3= Clause(literals3)
# clasues.append(clause3)

# testcase2
clasues.append(Clause({'D':False, 'B':False, 'C':True}))
clasues.append(Clause({'B':True, 'A':False, 'C':False}))
clasues.append(Clause({'B':False, 'E':True, 'C':False}))
clasues.append(Clause({'B':True, 'E':True, 'D':False}))
clasues.append(Clause({'B':True, 'E':True, 'C':False}))
clasues.append(Clause({'E':False, 'F':True, 'A':False}))
clasues.append(Clause({'E':True, 'F':False, 'A':True}))

for c in clasues:
    c.print()

p = 0.5
maxz_flip = 8
model = WALKSAT(clasues, p, maxz_flip)
print('final model:', model)


```

    {'D': False, 'B': False, 'C': True}
    {'B': True, 'A': False, 'C': False}
    {'B': False, 'E': True, 'C': False}
    {'B': True, 'E': True, 'D': False}
    {'B': True, 'E': True, 'C': False}
    {'E': False, 'F': True, 'A': False}
    {'E': True, 'F': False, 'A': True}
    symbols: {'F', 'B', 'C', 'A', 'D', 'E'}
    inital model: {'F': True, 'B': False, 'C': False, 'A': True, 'D': True, 'E': False}
    0  iteration: {'F': True, 'B': True, 'C': False, 'A': True, 'D': True, 'E': False}
    1  iteration: {'F': True, 'B': True, 'C': True, 'A': True, 'D': True, 'E': False}
    2  iteration: {'F': True, 'B': True, 'C': False, 'A': True, 'D': True, 'E': False}
    3  iteration: {'F': True, 'B': True, 'C': True, 'A': True, 'D': True, 'E': False}
    4  iteration: {'F': True, 'B': True, 'C': True, 'A': True, 'D': True, 'E': True}
    final model: {'F': True, 'B': True, 'C': True, 'A': True, 'D': True, 'E': True}

