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
    








    ################################################


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
    pass  # replace this with your code




######

def test():


    t = build_balanced(4)

    if t.size != 15 or t.depth() != 3:
        print("4 - error in build_balanced")

    if lowest_common_ancestor(t, 4, 7).key != 4 or lowest_common_ancestor(t, 2, 12).key != 8:
        print("4 - error in lowest_common_ancestor or build_balanced")

    '''if subtree_sum(t, 6) != 18 or subtree_sum(t, 12) != 84:
        print("4 - error in subtree_sum")
    '''

test()