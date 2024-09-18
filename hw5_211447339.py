# Skeleton file for HW5 - Winter 2024 - extended intro to CS

# Add your implementation to this file

# You may add other utility functions to this file,
# but you may NOT change the signature of the existing ones.

# Change the name of the file to include your ID number (hw5_ID.py).
import math
import random


#####################################
# Linked List   (code from lecture) #
#####################################

class Node:
    def __init__(self, val):
        self.value = val
        self.next = None

    def __repr__(self):
        return str(self.value)


class Linked_list:
    def __init__(self, seq=None):
        self.head = None
        self.size = 0
        if seq != None:
            for x in seq[::-1]:
                self.add_at_start(x)

    def __repr__(self):
        out = ""
        p = self.head
        while p != None:
            out += p.__repr__() + ", "
            p = p.next
        return "[" + out[:-2] + "]"  # discard the extra ", " at the end

    def add_at_start(self, val):
        ''' add node with value val at the list head '''
        tmp = self.head
        self.head = Node(val)
        self.head.next = tmp
        self.size += 1

    def __len__(self):
        ''' called when using Python's len() '''
        return self.size

    def index(self, val):
        ''' find index of (first) node with value val in list
            return None of not found '''
        p = self.head
        i = 0  # we want to return the location
        while p != None:
            if p.value == val:
                return i
            else:
                p = p.next
                i += 1
        return None  # in case val not found

    def __getitem__(self, i):
        ''' called when reading L[i]
            return value of node at index 0<=i<len '''
        assert 0 <= i < len(self)
        p = self.head
        for j in range(0, i):
            p = p.next
        return p.value

    def __setitem__(self, i, val):
        ''' called when using L[i]=val (indexing for writing)
            assigns val to node at index 0<=i<len '''
        assert 0 <= i < len(self)
        p = self.head
        for j in range(0, i):
            p = p.next
        p.value = val
        return None

    def insert(self, i, val):
        ''' add new node with value val before index 0<=i<=len '''
        assert 0 <= i <= len(self)
        if i == 0:
            self.add_at_start(val)  # makes changes to self.head
        else:
            p = self.head
            for j in range(0, i - 1):  # get to position i-1
                p = p.next
            # now add new element
            tmp = p.next
            p.next = Node(val)
            p.next.next = tmp
            self.size += 1

    def append(self, val):
        self.insert(self.size, val)

    def pop(self, i):
        ''' delete element at index 0<=i<len '''
        assert 0 <= i < len(self)
        if i == 0:
            self.head = self.head.next  # bypass first element
        else:  # i >= 1
            p = self.head
            for j in range(0, i - 1):
                p = p.next

            # now p is the element BEFORE index i
            p.next = p.next.next  # bypass element at index i

        self.size -= 1


##############################################
# Binary Search Tree     (code from lecture) #
##############################################

class TreeNode():
    def __init__(self, key, val=None):
        self.key = key
        self.val = val
        self.left = None
        self.right = None


    def __repr__(self):
        return "(" + str(self.key) + ":" + str(self.val) + ")"


class BinarySearchTree():
    def __init__(self, root=None, size=0):
        self.root = root
        self.size = size


    def __repr__(self):  # you don't need to understand the implementation of this method
        def printree(root):
            if not root:
                return ["#"]

            root_key = str(root.key)
            left, right = printree(root.left), printree(root.right)

            lwid = len(left[-1])
            rwid = len(right[-1])
            rootwid = len(root_key)

            result = [(lwid + 1) * " " + root_key + (rwid + 1) * " "]

            ls = len(left[0].rstrip())
            rs = len(right[0]) - len(right[0].lstrip())
            result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "\\" + rs * "_" + (rwid - rs) * " ")

            for i in range(max(len(left), len(right))):
                row = ""
                if i < len(left):
                    row += left[i]
                else:
                    row += lwid * " "
                    
                row += (rootwid + 2) * " "

                if i < len(right):
                    row += right[i]
                else:
                    row += rwid * " "

                result.append(row)

            return result

        return '\n'.join(printree(self.root))


    def lookup(self, key):
        ''' return value of node with key if exists, else None '''
        node = self.root
        while node != None:
            if key == node.key:
                return node.val  # found!
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None


    def insert(self, key, val):
        ''' insert node with key,val into tree.
            if key already there, just update its value '''

        parent = None  # this will be the parent of the new node
        node = self.root

        while node != None:  # keep descending the tree
            if key == node.key:
                node.val = val  # update the val for this key
                return

            parent = node
            if key < node.key:
                node = node.left
            else:
                node = node.right

        if parent == None:  # was empty tree, need to update root
            self.root = TreeNode(key, val)
        elif key < parent.key:
            parent.left = TreeNode(key, val)  # "hang" new node as left child
        else:
            parent.right = TreeNode(key, val)  # "hang"    ...     right child

        self.size += 1
        return None


    def minimum(self):
        ''' return value of node with minimal key '''

        if self.root == None:
            return None  # empty tree has no minimum
        node = self.root
        while node.left != None:
            node = node.left
        return node.val


    def depth(self):
        ''' return depth of tree, uses recursion '''
        def depth_rec(node):
            if node == None:
                return -1
            else:
                return 1 + max(depth_rec(node.left), depth_rec(node.right))

        return depth_rec(self.root)
##############
# QUESTION 1 #
##############
class LLLNode:
    def __init__(self, val):
        self.next_list = []
        self.val = val

    def __repr__(self):
        st = "Value: " + str(self.val) + "\n"
        st += "Neighbors:" + "\n"
        for p in self.next_list:
            st += " - Node with value: " + str(p.val) + "\n"
        return st[:-1]


class LogarithmicLinkedList:
    def __init__(self):
        self.head = None
        self.len = 0

    def __len__(self):
        return self.len

    def add_at_start(self, val):
        node = LLLNode(val)
        if len(self) == 0:
            self.head = node
            self.len = 1
            return None
        
        # Add your code here #
        else:
            node.next_list = [self.head]
            current = self.head
            i = 0

            while i < len(self) and len(current.next_list) > i:
                node.next_list.append(current.next_list[i])
                current = current.next_list[i]
                i += 1

        self.head = node
        self.len += 1
            
        return None


    def __getitem__(self, i):

        def int_to_binary_string(n):
            bin_repr = bin(n)[2:]
            return str(bin_repr)
        
        if i < 0 or i >= self.len:
           raise IndexError("Index out of range")
    
        current = self.head #start at the head of the linkedList
        #now we want to visit the 
        binary = int_to_binary_string(i)
        binary_flipped = binary[::-1]
        
        for bit in binary_flipped:
            if bit == '1':
                index = len(current.next_list) - 1
                current = current.next_list[index]

        return current.val
    # Optional - improve this code!
    def __contains__(self, val):
        p = self.head
        k = 1
        while k != 0:
            if p.val == val:
                return True
            k = 0
            m = len(p.next_list)
            while k < m and p.next_list[k].val <= val:
                k += 1
            if k > 0:
                p = p.next_list[k - 1]
        return False


##############
# QUESTION 2 #
##############

def gen1():
    s = 0
    while True:
        for i in range(-s, s + 1):
            yield i, s - abs(i) if i <= 0 else - (s - abs(i))
        
        s += 1



def gen2(g):
    total = 0
    for num in g:
        total += num
        yield total


def gen3(g):
    for num in g:
        if num > 0:
            yield num  


def gen4(g):
    prev_value = None
    first_diff = None
    prev_diff = None
    state = None

    for value in g:
        if state == False:
            yield False
        elif prev_value is None: #this is the first value
            prev_value = value
            yield True
        else:
            if prev_diff is None: #this is the second value, theres no way to decide 
                diff = value - prev_value
                prev_diff = diff
                if value - prev_value > 0:
                    first_diff =  'increasing'
                elif value - prev_value < 0:
                    first_diff = 'decreasing'
                else:
                    first_diff = 'equal'
            else: #this is after the third instance is produced
                if first_diff == 'equal': 
                    yield True
                elif first_diff == 'increasing':
                    if value <= prev_value:
                        yield False
                        state = False
                        
                    else:
                        yield True
                elif first_diff == 'decreasing':
                    if value >= prev_value:
                        yield False
                        state = False
                    else:
                        yield True
                            
        prev_value = value
        if state == False:
            yield False


def gen5(g1, g2):
    pass  # replace this with your code (or don't, if there does not exist such generator with finite delay)


def gen6(g1, g2):
    pass  # replace this with your code (or don't, if there does not exist such generator with finite delay)

def gen7():
    pass  # replace this with your code (or don't, if there does not exist such generator with finite delay)

##############
# QUESTION 3 #
##############

def lowest_common_ancestor(t, n1, n2):
    def helper(root, n1, n2):

        if not root:
            return None
        #if both keys are smaller than the root, then the wanted lies left the subtree
        if n1 < root.key and n2 < root.key:
            return helper(root.left, n1, n2)
        
        #if both nodes are larger than the root, the wanted lies in a right down, which means

        if n1 > root.key and n2 > root.key:
            return helper(root.right, n1, n2)
        
        #if one key is smaller and the other key is larger(or matches roots key)), then root is the wanted

        return root
    return helper(t.root, n1, n2)



def build_balanced(n):
    def build_recursive(start, end, depth):
        if depth == 0 or start > end:
            return None
        
        mid = (start + end) // 2
        root = TreeNode(mid)
        
        root.left = build_recursive(start, mid - 1, depth - 1)
        root.right = build_recursive(mid + 1, end, depth - 1)
        
        return root
    
    if n <= 0:
        return BinarySearchTree()
    
    num_nodes = 2**n -1
    root = build_recursive(1, num_nodes, n)
    return BinarySearchTree(root, num_nodes)



def subtree_sum(t, k):
    def subtree_sum_helper(node):
        #lets write the base case for the function: if the next node is none, we add the key to the sum
        if node is None:
            return 0
        return node.key + subtree_sum_helper(node.left) + subtree_sum_helper(node.right)
    
    #manuually search for the node with the k value:
    current = t.root
    while current is not None:
        if k == current.key:
            return subtree_sum_helper(current)
        elif k < current.key:
            current = current.left
        else:
            current = current.right
    return 0


    

##############
# QUESTION 5 #
##############
def prefix_suffix_overlap(lst, k):
    list = []
    for i in range(len(lst)): #itetrate by number on the lst
        elem_pre = lst[i] # name elem_pre by this so we dont call it by list iteratition every time
        prefix = elem_pre[:k] #name the prefix by indexing
        for j in range(len(lst)):  #itetrate by number on the lst
            elem_suf = lst[j]
            if j != i: 
                if elem_suf[-k:] == prefix:
                    list.append((i,j))
        
    return list

class Dict:
    def __init__(self, m, hash_func=hash):
        """ initial hash table, m empty entries """
        self.table = [[] for i in range(m)]
        self.hash_mod = lambda x: hash_func(x) % m

    def __repr__(self):
        L = [self.table[i] for i in range(len(self.table))]
        return "".join([str(i) + " " + str(L[i]) + "\n" for i in range(len(self.table))])

    def insert(self, key, value):
        """ insert key,value into table
            Allow repetitions of keys """
        i = self.hash_mod(key)  # hash on key only
        item = [key, value]  # pack into one item
        self.table[i].append(item)

    def find(self, key):
        """ returns ALL values of key as a list, empty list if none """
        i = self.hash_mod(key)
        lst = [item[1] for item in self.table[i] if item[0] == key]
        return lst
    
        
def prefix_suffix_overlap_hash1(lst, k):
    #we need to create now a hashtable, 
    # it hashes the key of the element we are working at at the moment
    #and saves the element in the index i, we need to initiate the hashtable first, m-elements
    # and hash functions

    hashtable = Dict(len(lst)) #we created the hash
    for i in range(len(lst)): #create a loop which iterates over the elmts of lst
        hashtable.insert(lst[i][:k], i)
        #now after creating the hash table, we need to iterate over the suffixs of the 
        # elems in the list and by the hashed suffix, check if the hash is in the hashtable,
        #  if yes, access the id in the table, iterate ovwer the lists inside it,
        # with the list containing the second part of the lists
    result = []
    for j in range(len(lst)):
        matching_indexes = hashtable.find(lst[j][-k:])
        
        for i in matching_indexes:
            if i != j:
                result.append((i,j))

    return result




##############
# QUESTION 6 #
##############

def int_size(N):
    return math.floor(math.log(N,2)) + 1


def maxmatch(T, p, W=2**12-1, L=2**5-1):
    assert isinstance(T,str)
    n = len(T)
    m = 0
    k = 0
    for offset in range(1, 1+min(p, W)):
        match_len = 0
        j = p-offset
        while match_len < min(n-p, L) and T[j+match_len] == T[p+match_len]:
            match_len+=1
        if match_len > k:
            k = match_len
            m = offset
    return m, k


# Modify this code #
def LZW_compress_v2(text, c, W=2**12-1, L=2**5-1):
    intermediate = []
    n = len(text)
    p = 0
    rep_len = int_size(W) + int_size(L) + 1

    while p < n:
        m, k = maxmatch(text, p, W, L)
        if 8 * k <= rep_len:
            # Directly add the character as is
            intermediate.append(text[p])
            p += 1
        else:
            # Add the LZW encoded [m, k] as is
            intermediate.append([m, k])
            p += k
    
    return intermediate


# Modify this code #
def inter_to_bin_v2(intermediate, c, W=2**12-1, L=2**5-1):
    W_width = int_size(W)
    L_width = int_size(L)
    bits = []
    for elem in intermediate:
        if type(elem) == str:
            bits.append("0")
            bits.append((bin(ord(elem))[2:]).zfill(7))
        else:
            bits.append("1")
            m, k = elem
            bits.append((bin(m)[2:]).zfill(W_width))
            bits.append((bin(k)[2:]).zfill(L_width))
    return "".join(ch for ch in bits)


# This does not require any changes #
def LZW_decompress(intermediate):
    text_lst = []
    for i in range(len(intermediate)):
        if type(intermediate[i]) == str:
            text_lst.append(intermediate[i])
        else:
            m, k = intermediate[i]
            for j in range(k):
                text_lst.append(text_lst[-m])
    return "".join(text_lst)


##############
# TESTS      #
##############
def test():
    # Q1
    lst = LogarithmicLinkedList()
    lst.add_at_start(1)
    lst.add_at_start("hello")
    lst.add_at_start(True)
    if lst[0] != True or len(lst) != 3:
        print("1 - error in LogarithmicLinkedList")

    # Q4
    t = build_balanced(4)

    if t.size != 15 or t.depth() != 3:
        print("4 - error in build_balanced")

    if lowest_common_ancestor(t, 4, 7).key != 4 or lowest_common_ancestor(t, 2, 12).key != 8:
        print("4 - error in lowest_common_ancestor or build_balanced")

    if subtree_sum(t, 6) != 18 or subtree_sum(t, 12) != 84:
        print("4 - error in subtree_sum")
    # Q5
    lst = ["abcd", "cdab", "aaaa", "bbbb", "abff"]
    k = 2
    if sorted(prefix_suffix_overlap(lst, k)) != sorted([(0, 1), (1, 0), (4, 1)]):
        print("5 - error in prefix_suffix_overlap")

    d = Dict(3)
    d.insert("a", 56)
    d.insert("a", 34)
    if sorted(d.find("a")) != sorted([56, 34]) or d.find("b") != []:
        print("5 - error in Dict.find")

    lst = ["abcd", "cdab", "aaaa", "bbbb", "abff"]
    k = 2
    if sorted(prefix_suffix_overlap_hash1(lst, k)) != sorted([(0, 1), (1, 0), (4, 1)]):
        print("5 - error in prefix_suffix_overlap_hash1")


    # Q6
    c = {'a': '0', 'b': '10', 'c': '110', 'd': '1110', 'e': '1111'}
    W = 2**5-1
    L = 2**3-1
    if LZW_compress_v2("abcdeabccde", c, W, L) != ['a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', [6, 3]] or \
            LZW_compress_v2("ededaaaaa", c, W, L) != ['e', 'd', [2, 2], 'a', 'a', 'a', 'a', 'a']:
        print("Error in Q3b - LZW_compress_v2")
    if inter_to_bin_v2(['e', 'd', [2, 2]], c, W, L) != "0111101110100010010":
        print("Error in Q3b - inter_to_bin_v2")

test()
