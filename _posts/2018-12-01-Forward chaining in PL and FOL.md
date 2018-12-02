---  
tag: AI 
---

# forward chaining

## In Propositional Logic
PL-FC-ENTAILS(KB, q):    
KB: a set of definite caluses    
q: a symbol    

- begins from known facts in KB    
    if all the premises of an implication are known,    
    then its conclusion is added to the set of know facts     
- continue until the q is added or no further inferences can be made. 


PL-FC-ENTAILS(KB, q): return True of False    
    
    count = a table, where count[c] is the number of symbols in c's premise.    
    infered = a table, where inferred[s] is initially false for all symbols.    
    agenda = a queue of symbols, initially symbols known to be true in KB.
    
    while agenda is not empty do:
        p = pop(agenda)    
        if p==q, then return true    
        if inferrend[p] = false:
            inferrend[p] = true
            for each clause in c in KB where p is in c.PREMISE do:
                 drease count[c]
                 if count[c] == 0:
                     add c.CONCLUSION to agenda
    return False

## In First-Order Logic

FOL-FC-ASK(KB, query): return a substitution or false 

    repeat until newInfered is empty
        newInfered = {} # the new sentences infered on each iteration
        for each rule r in KB do:
            (p1 and ...and pn => q) <- STANDARDIZE-APART(r)
            for each ø such that (p1 and ...and pn)ø == (p1' and ...and pn')ø:
                q' = SUBST(ø, q)
                if q' is not a renaming of a sentence already in KB or newInfered:
                    add q' to newInfered
                    ret = UNIFY(q', query)
                    if ret is not fail:
                        return ø
        add newInfered into KB
    return False

# Implementation in FOL


```python
class Atom():
    
    def __init__(self, ops, args):
        self.ops = ops
        self.args = args
    def getOps(self):
        return self.ops
    def getArgs(self):
        return self.args
    def __str__(self):
        return self.ops+"("+ self.args+ ")"
class Implication():
    
    def __init__(self, premise, conclusion):
        self.premise = premise
        self.conclusion = conclusion
    def getSentences(self):
        ret = []
        ret = [s for s in self.premise]
        ret.append(self.conclusion)
        return ret 
    def __str__(self):
        ret = ""
        for atom in self.premise:
            ret += str(atom) + " && "
        ret = ret[0:-3] + "==> " + str(self.conclusion)
        return ret

class KB():
    def __init__(self, rules, facts):
        self.rules = rules # a list
        self.facts = facts # a list
    
    def getKB(self):
        return rules+facts
    def addFact(self, fact):
        self.facts.append(fact)
    
    def __str__(self):
        ret = ""
        for r in self.rules:
            ret += str(r) +'\n'
        ret += '\n'
        for f in self.facts:
            ret += str(f) +'\n'
        return ret
        
```


```python
flatten = lambda l: [item for sublist in l for item in sublist]

def STANDARDIZE(rules):
    '''
    eliminate overlap variables among rules
    '''
    delimer = ',' 
    count = 0
    for rule in rules:
        # American(x) Weapon(y) Sells(x,y,z) Hostile(z) Criminal(x)
        sentences = rule.getSentences()
        variables_constants = set(flatten([s.args.split(delimer) for s in sentences]))
        variables = []
        for item in variables_constants:
            if VARIABLE(item):
                variables.append(item)
                
        new_variables = []
        for i in range(len(variables)):
            new_variables.append("x"+str(count))
            count += 1
            
        for s in sentences:
            news = []
            olds = s.args.split(delimer)
            for old in olds:
                if VARIABLE(old):
                    idx = variables.index(old)
                    new = new_variables[idx]
                    news.append(new)
                else:
                    news.append(old)
            args = delimer.join(news)
            s.args = args

def oneVariable(x):
    if (not isinstance(x, Atom)) and (not isinstance(x, list)):
        return True
    
    return False

def UNIFY(p1, p2, theta):
    '''
    get substition of p1, and p2 such that p1 theta = p2 theta
    p1,p2: variable(different from variable in KB), constant list, Atom
    
    theta: the substition built up so far, defalut to empty
    '''
    delimer = ','
    if theta == -1:# -1 means failure
        return -1
    
#     print('p1, p2:', p1, p2)
    if (str(p1) == str(p2)):
        return theta
    
    if(oneVariable(p1)):
#         print('-- variable')
        return UNIFY_VAR(p1, p2, theta)
    
    if (isinstance(p1, Atom) and isinstance(p2, Atom)):
#         print('-- atom')
#         return UNIFY(p1.ops, p2.ops, UNIFY(p1.args.split(delimer), p2.args.split(delimer), theta))
        if(p1.ops != p2.ops):
            return -1
        return UNIFY(p1.args.split(delimer), p2.args.split(delimer), theta)
    
    if (isinstance(p1, list) and isinstance(p2, list)):
#         print('-- list')
        if len(p1) != len(p2):
            return -1
        return UNIFY(p1, p2, UNIFY(p1.pop(0), p2.pop(0), theta))
    return -1
        
def UNIFY_VAR(p1, p2, theta):
#     print(p1, p2, theta)
    val = theta.get(p1, -1)
    if(val != -1):
        return UNIFY(val, p2, theta)
    val = theta.get(p2, -1)
    if(val != -1):
        return UNIFY(p1, val, theta)
    
    # TODO
    # OCCUR-CHECK for p1 is a variable, p2 is an Atom, if p1 in P2.args, return -1
    
    theta[p1] = p2
    return theta

def isIn(q, kb):
    '''
    check wether q is already in kb
    '''
    for fact in kb.facts:
        if (str(fact) == str(q)):
            print('in', fact)
            return True
    return False
    
def SUBST(theta, q):
    delimer = ','
    args = q.args.split(delimer)
    news =[]
    for old in args:
        new = theta.get(old, -1)
        if(new == -1):
            news.append(old)
        else:
            news.append(new)
    args = delimer.join(news)   
    
    p = Atom(q.ops, args)
#     print('SUBST:', q, p)
    return p
```


```python
def FOL_FC_ASK(kb, query):
    DEBUG = True
    print('----STANDARDIZE---')
    STANDARDIZE(kb.rules)
#     if DEBUG: print(kb)

    print('----ITERATION---')
    i = 0
    while(True):
        
        if DEBUG: print('ITERATION: ', i)
        newInfered = []
        facts = kb.facts
        all_kb = kb.getKB()
        
        for r in kb.rules:
            print('rule:', r)
            # matching
            # note: it is expensive
            premise = r.premise
            theta = {}
            satisfied = True
            for p in premise:
                satisfied_p = False
                for s in all_kb:
                    ret = UNIFY(p,s, theta)
                    if(ret != -1):
                        satisfied_p = True
                if satisfied_p != True:
                    satisfied = False
            if(satisfied):
                print(theta)
                q_prim = SUBST(theta, r.conclusion)
                
                ret = UNIFY(q_prim, query, {})
                if(ret != -1):
                    return theta
                #check whether q_prim is already in KB or newInfered in some format
                in_kb = isIn(q_prim, kb)
                if(in_kb != True):
                    newInfered.append(q_prim)  
            else:
                print('unsatisfied!!!')
        print()        
        #add newInfered to KB
        for fact in newInfered:
            print('new fact:', fact)
            kb.addFact(fact)
#         print(kb)
        
        print()
        i += 1
        if(len(newInfered) == 0):
            return -1 #  means Failure
```

# Evaluation


```python
# Build KB

atom1 = Atom('American', 'x') # variable starts with lowercase
atom2 = Atom('Weapon', 'y')
atom3 = Atom('Sells', 'x,y,z') # with no space
atom4 = Atom('Hostile', 'z')
conclusion = Atom('Criminal', 'x')
premise = [atom1, atom2, atom3, atom4]
rule1 = Implication(premise, conclusion)

atom21 = Atom('Missile', 'x')
atom22 = Atom('Owns', 'Nono,x')
conclusion = Atom('Sells', 'West,x,Nono')
premise = [atom21, atom22]
rule2 = Implication(premise, conclusion)

atom31 = Atom('Missile', 'x')
conclusion = Atom('Weapon', 'x')
premise = [atom31]
rule3 = Implication(premise, conclusion)

atom41 = Atom('Enemy', 'x,America')
conclusion = Atom('Hostile', 'x')
premise = [atom41]
rule4 = Implication(premise, conclusion)

fact1 = Atom('American', 'West')# Constant starts with upercase
fact2 = Atom('Owns', 'Nono,M1')# Constant starts with upercase
fact3 = Atom('Missile', 'M1')# Constant starts with upercase
fact4 = Atom('Enemy', 'Nono,America')# Constant starts with upercase

rules = [rule1, rule2, rule3, rule4]
facts = [fact1, fact2, fact3, fact4]

# for test
# print(atom22, fact2)
# theta = UNIFY(atom22,fact2, {})    
# print(theta)

kb = KB(rules, facts)
print(kb)
```

    American(x) && Weapon(y) && Sells(x,y,z) && Hostile(z) ==> Criminal(x)
    Missile(x) && Owns(Nono,x) ==> Sells(West,x,Nono)
    Missile(x) ==> Weapon(x)
    Enemy(x,America) ==> Hostile(x)
    
    American(West)
    Owns(Nono,M1)
    Missile(M1)
    Enemy(Nono,America)
    



```python
query = Atom('Criminal', 'West')
theta = FOL_FC_ASK(kb, query)
print()
print('----Final-----')
if(theta != -1):
    print(query, 'is True.')
    print(theta)
else:
    print('Fail to infer')
print()
```

    ----STANDARDIZE---
    ----ITERATION---
    ITERATION:  0
    rule: American(x1) && Weapon(x2) && Sells(x1,x2,x0) && Hostile(x0) ==> Criminal(x1)
    unsatisfied!!!
    rule: Missile(x3) && Owns(Nono,x3) ==> Sells(West,x3,Nono)
    {'x3': 'M1'}
    rule: Missile(x4) ==> Weapon(x4)
    {'x4': 'M1'}
    rule: Enemy(x5,America) ==> Hostile(x5)
    {'x5': 'Nono'}
    
    new fact: Sells(West,M1,Nono)
    new fact: Weapon(M1)
    new fact: Hostile(Nono)
    
    ITERATION:  1
    rule: American(x1) && Weapon(x2) && Sells(x1,x2,x0) && Hostile(x0) ==> Criminal(x1)
    {'x1': 'West', 'x2': 'M1', 'x0': 'Nono'}
    
    ----Final-----
    Criminal(West) is True.
    {'x1': 'West', 'x2': 'M1', 'x0': 'Nono'}
    

