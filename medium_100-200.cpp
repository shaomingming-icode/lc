#include "medium.h"

---------------------------------------------------------------------

//102 Binary Tree Level Order Traversal 二叉树层序遍历
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
    Given binary tree [3,9,20,null,null,15,7],
        3
       / \
      9  20
        /  \
       15   7
    return its level order traversal as:
    [
      [3],
      [9,20],
      [15,7]
    ]

vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    vector<vector<int>> res;
    queue<TreeNode*> q{{root}};
    while (!q.empty()) {
        vector<int> oneLevel;
        for (int i = q.size(); i > 0; --i) {
            TreeNode *t = q.front(); q.pop();
            oneLevel.push_back(t->val);
            if (t->left) q.push(t->left);
            if (t->right) q.push(t->right);
        }
        res.push_back(oneLevel);
    }
    return res;
}

---------------------------------------------------------------------

//103 Binary Tree Zigzag Level Order Traversal 二叉树的之字形层序遍历
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
        3
       / \
      9  20
        /  \
       15   7
    return its zigzag level order traversal as:
    [
      [3],
      [20,9],
      [15,7]
    ]

vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    if (!root) return {};
    vector<vector<int>> res;
    queue<TreeNode*> q{{root}};
    int cnt = 0;
    while (!q.empty()) {
        vector<int> oneLevel;
        for (int i = q.size(); i > 0; --i) {
            TreeNode *t = q.front(); q.pop();
            oneLevel.push_back(t->val);
            if (t->left) q.push(t->left);
            if (t->right) q.push(t->right);
        }
        if (cnt % 2 == 1) reverse(oneLevel.begin(), oneLevel.end());
        res.push_back(oneLevel);
        ++cnt;
    }
    return res;
}

---------------------------------------------------------------------

//105 Construct Binary Tree from Preorder and Inorder Traversal 由先序和中序遍历建立二叉树
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

    preorder = [3,9,20,15,7]
    inorder = [9,3,15,20,7]
Return the following binary tree:
        3
       / \
      9  20
        /  \
       15   7

由于先序的顺序的第一个肯定是根，所以原二叉树的根节点可以知道，题目中给了一个很关键的条件就是树中没有相同元素，有了这个条件我们就可以在中序遍历中也定位出根节点的位置，并以根节点的位置将中序遍历拆分为左右两个部分，分别对其递归调用原函数

TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
    return buildTree(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
}

TreeNode *buildTree(vector<int> &preorder, int pLeft, int pRight, vector<int> &inorder, int iLeft, int iRight) {
    if (pLeft > pRight || iLeft > iRight) return NULL;
    int i = 0;
    for (i = iLeft; i <= iRight; ++i) {
        if (preorder[pLeft] == inorder[i]) break;
    }
    TreeNode *cur = new TreeNode(preorder[pLeft]);
    cur->left = buildTree(preorder, pLeft + 1, pLeft + i - iLeft, inorder, iLeft, i - 1);
    cur->right = buildTree(preorder, pLeft + i - iLeft + 1, pRight, inorder, i + 1, iRight);
    return cur;
}

---------------------------------------------------------------------

//106 Construct Binary Tree from Inorder and Postorder Traversal 由中序和后序遍历建立二叉树
Given inorder and postorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given
    inorder = [9,3,15,20,7]
    postorder = [9,15,7,20,3]
Return the following binary tree:

        3
       / \
      9  20
        /  \
       15   7

由于后序的顺序的最后一个肯定是根，所以原二叉树的根节点可以知道，题目中给了一个很关键的条件就是树中没有相同元素，有了这个条件我们就可以在中序遍历中也定位出根节点的位置，并以根节点的位置将中序遍历拆分为左右两个部分，分别对其递归调用原函数

TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
    return buildTree(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
}

TreeNode *buildTree(vector<int> &inorder, int iLeft, int iRight, vector<int> &postorder, int pLeft, int pRight) {
    if (iLeft > iRight || pLeft > pRight) return NULL;
    TreeNode *cur = new TreeNode(postorder[pRight]);
    int i = 0;
    for (i = iLeft; i < inorder.size(); ++i) {
        if (inorder[i] == cur->val) break;
    }
    cur->left = buildTree(inorder, iLeft, i - 1, postorder, pLeft, pLeft + i - iLeft - 1);
    cur->right = buildTree(inorder, i + 1, iRight, postorder, pLeft + i - iLeft, pRight - 1);
    return cur;
}

为什么不能由先序和后序遍历建立二叉树呢，这是因为先序和后序遍历不能唯一的确定一个二叉树，比如下面五棵树：
    　1　　　　　preorder:　　  1　　2　　3
     / \　　　　 inorder:　　   2　　1　　3
    2    3　　   postorder:　　 2　　3　　1

 

        1   　　 preorder:　　  1　　2　　3
       / 　　　　inorder:　　   3　　2　　1
      2 　　     postorder: 　　3　　2　　1
     /
    3

      1　　　　  preorder:　　  1　　2　　3
     / 　　　　  inorder:　　   2　　3　　1
    2 　　　　　 postorder:　　 3　　2　　1
     \
      3

    1　　　　    preorder:　　  1　　2　　3
     \ 　　　    inorder:　　   1　　3　　2
      2 　　　　 postorder:　　 3　　2　　1
     /
    3

    1　　　      preorder:　　  1　　2　　3
     \ 　　　　　inorder:　　   1　　2　　3
      2 　　　　 postorder:　　 3　　2　　1
       \
　　　　3

从上面可以看出，对于先序遍历都为1 2 3的五棵二叉树，它们的中序遍历都不相同，而它们的后序遍历却有相同的，所以只有和中序遍历一起才能唯一的确定一棵二叉树
   
---------------------------------------------------------------------

//109 Convert Sorted List to Binary Search Tree 将有序链表转为二叉搜索树
Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example:
    Given the sorted linked list: [-10,-3,0,5,9],

    One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

          0
         / \
       -3   9
       /   /
     -10  5

找到中点后，要以中点的值建立一个数的根节点，然后需要把原链表断开，分为前后两个链表，都不能包含原中节点，然后再分别对这两个链表递归调用原函数，分别连上左右子节点即可

TreeNode *sortedListToBST(ListNode* head) {
    if (!head) return NULL;
    if (!head->next) return new TreeNode(head->val);
    ListNode *slow = head, *fast = head, *last = slow;
    while (fast->next && fast->next->next) {
        last = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    fast = slow->next;
    last->next = NULL;
    TreeNode *cur = new TreeNode(slow->val);
    if (head != slow) cur->left = sortedListToBST(head);
    cur->right = sortedListToBST(fast);
    return cur;
}

---------------------------------------------------------------------

//113 Path Sum II 二叉树路径之和之二
Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Note: A leaf is a node with no children.

Example:
    Given the below binary tree and sum = 22,

          5
         / \
        4   8
       /   / \
      11  13  4
     /  \    / \
    7    2  5   1
    Return:
    [
       [5,4,11,2],
       [5,8,4,5]
    ]

vector<vector<int> > pathSum(TreeNode *root, int sum) {    
    vector<vector<int>> res;
    vector<int> out;
    helper(root, sum, out, res);
    return res;
}

void helper(TreeNode* node, int sum, vector<int>& out, vector<vector<int>>& res) {
    if (!node) return;
    out.push_back(node->val);
    if (sum == node->val && !node->left && !node->right) {
        res.push_back(out);
    }
    helper(node->left, sum - node->val, out, res);
    helper(node->right, sum - node->val, out, res);
    out.pop_back();
}

---------------------------------------------------------------------

//114 Flatten Binary Tree to Linked List 将二叉树展开成链表
Given a binary tree, flatten it to a linked list in-place.

For example, given the following tree:
        1
       / \
      2   5
     / \   \
    3   4   6
    The flattened tree should look like:
    1
     \
      2
       \
        3
         \
          4
           \
            5
             \
              6

从根节点开始出发，先检测其左子结点是否存在，如存在则将根节点和其右子节点断开，将左子结点及其后面所有结构一起连到原右子节点的位置，把原右子节点连到元左子结点最后面的右子节点之后

void flatten(TreeNode *root) {
    TreeNode *cur = root;
    while (cur) {
        if (cur->left) {
            TreeNode *p = cur->left;
            while (p->right) p = p->right;
            p->right = cur->right;
            cur->right = cur->left;
            cur->left = NULL;
        }
        cur = cur->right;
    }
}

void flatten(TreeNode* root) {
    if (!root) return;
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        TreeNode *t = s.top(); s.pop();
        if (t->left) {
            TreeNode *r = t->left;
            while (r->right) r = r->right;
            r->right = t->right;
            t->right = t->left;
            t->left = NULL;
        }
        if (t->right) s.push(t->right);
    }
}

树的遍历有递归和非递归两种方法

void flatten(TreeNode *root) {
    if (!root) return;
    if (root->left) flatten(root->left);
    if (root->right) flatten(root->right);
    TreeNode *tmp = root->right;
    root->right = root->left;
    root->left = NULL;
    while (root->right) root = root->right;
    root->right = tmp;
}
    
---------------------------------------------------------------------

//116 Populating Next Right Pointers in Each Node 每个节点的右向指针

You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

实际上是树的层序遍历的应用

递归
Node* connect(Node* root) {
    if (!root) return NULL;
    if (root->left) root->left->next = root->right;
    if (root->right) root->right->next = root->next? root->next->left : NULL;
    connect(root->left);
    connect(root->right);
    return root;
}

非递归
Node* connect(Node* root) {
    if (!root) return NULL;
    queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            Node *t = q.front(); q.pop();
            if (i < size - 1) {
                t->next = q.front();
            }
            if (t->left) q.push(t->left);
            if (t->right) q.push(t->right);
        }
    }
    return root;
}

两个指针 start 和 cur，其中 start 标记每一层的起始节点，cur 用来遍历该层的节点
Node* connect(Node* root) {
    if (!root) return NULL;
    Node *start = root, *cur = NULL;
    while (start->left) {
        cur = start;
        while (cur) {
            cur->left->next = cur->right;
            if (cur->next) cur->right->next = cur->next->left;
            cur = cur->next;
        }
        start = start->left;
    }
    return root;
}

---------------------------------------------------------------------

//117 Populating Next Right Pointers in Each Node II 每个节点的右向指针之二
Given a binary tree

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

Note:

You may only use constant extra space.
Recursive approach is fine, implicit stack space does not count as extra space for this problem.

递归
Node* connect(Node* root) {
    if (!root) return NULL;
    Node *p = root->next;
    while (p) {
        if (p->left) {
            p = p->left;
            break;
        }
        if (p->right) {
            p = p->right;
            break;
        }
        p = p->next;
    }
    if (root->right) root->right->next = p; 
    if (root->left) root->left->next = root->right ? root->right : p; 
    connect(root->right);
    connect(root->left);
    return root;
}

迭代 非常量空间
Node* connect(Node* root) {
    if (!root) return NULL;
    queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        int len = q.size();
        for (int i = 0; i < len; ++i) {
            Node *t = q.front(); q.pop();
            if (i < len - 1) t->next = q.front();
            if (t->left) q.push(t->left);
            if (t->right) q.push(t->right);
        }
    }
    return root;
}

迭代 常量空间
建立一个dummy结点来指向每层的首结点的前一个结点
指针cur用来遍历这一层
Node* connect(Node* root) {
    Node *dummy = new Node(0, NULL, NULL, NULL), *cur = dummy, *head = root;
    while (root) {
        if (root->left) {
            cur->next = root->left;
            cur = cur->next;
        }
        if (root->right) {
            cur->next = root->right;
            cur = cur->next;
        }
        root = root->next;
        if (!root) {
            cur = dummy;
            root = dummy->next;
            dummy->next = NULL;
        }
    }
    return head;
}

---------------------------------------------------------------------

//120 Triangle 三角形
Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

For example, given the following triangle
    [
         [2],
        [3,4],
       [6,5,7],
      [4,1,8,3]
    ]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

Note:

Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.

修改原数组的DP方法
int minimumTotal(vector<vector<int>>& triangle) {
    for (int i = 1; i < triangle.size(); ++i) {
        for (int j = 0; j < triangle[i].size(); ++j) {
            if (j == 0) {
                triangle[i][j] += triangle[i - 1][j];
            }
            else if (j == triangle[i].size() - 1) {
                triangle[i][j] += triangle[i - 1][j - 1];
            }
            else {
                triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j]);
            }
        }
    }
    return *min_element(triangle.back().begin(), triangle.back().end());
}

复制三角形最后一行，作为用来更新的一位数组
int minimumTotal(vector<vector<int>>& triangle) {
    vector<int> dp(triangle.back());
    for (int i = (int)triangle.size() - 2; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j];
        }
    }
    return dp[0];
}

---------------------------------------------------------------------

//127 Word Ladder 词语阶梯
Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:
    Input:
    beginWord = "hit",
    endWord = "cog",
    wordList = ["hot","dot","dog","lot","log","cog"]

    Output: 5

Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.
Example 2:
    Input:
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log"]

    Output: 0

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.

int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (!wordSet.count(endWord)) return 0;
    queue<string> q{{beginWord}};
    int res = 0;
    while (!q.empty()) {
        for (int k = q.size(); k > 0; --k) {
            string word = q.front(); q.pop();
            if (word == endWord) return res + 1;
            for (int i = 0; i < word.size(); ++i) {
                string newWord = word;
                for (char ch = 'a'; ch <= 'z'; ++ch) {
                    newWord[i] = ch;
                    if (wordSet.count(newWord) && newWord != word) {
                        q.push(newWord);
                        wordSet.erase(newWord);
                    }
                }
            }
        }
        ++res;
    }
    return 0;
}

---------------------------------------------------------------------

//129 Sum Root to Leaf Numbers 求根到叶节点数字之和
Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.

Note: A leaf is a node with no children.

Example:
    Input: [1,2,3]
        1
       / \
      2   3
    Output: 25
Explanation:
    The root-to-leaf path 1->2 represents the number 12.
    The root-to-leaf path 1->3 represents the number 13.
    Therefore, sum = 12 + 13 = 25.

Example 2:
    Input: [4,9,0,5,1]
        4
       / \
      9   0
     / \
    5   1
    Output: 1026
Explanation:
    The root-to-leaf path 4->9->5 represents the number 495.
    The root-to-leaf path 4->9->1 represents the number 491.
    The root-to-leaf path 4->0 represents the number 40.
    Therefore, sum = 495 + 491 + 40 = 1026.

递归 容易理解
int sumNumbers(TreeNode* root) {
    int sum = 0;
    sumNumbersDFS(root, 0, &sum);
    return sum;
}

void sumNumbersDFS(TreeNode* root, int tempSum, int *sum) {
    if (!root) return;
    tempSum = tempSum * 10 + root->val;
    if (!root->left && !root->right) {
        *sum += tempSum;
    }
    sumNumbersDFS(root->left, tempSum, sum);
    sumNumbersDFS(root->right, tempSum, sum);
}

迭代 使用栈来实现深度优先搜索
int sumNumbers(TreeNode* root) {
    if (!root) return 0;
    int res = 0;
    stack<TreeNode*> st{{root}};
    while (!st.empty()) {
        TreeNode *t = st.top(); st.pop();
        if (!t->left && !t->right) {
            res += t->val;
        }
        if (t->right) {
            t->right->val += t->val * 10;
            st.push(t->right);
        }
        if (t->left) {
            t->left->val += t->val * 10;
            st.push(t->left);
        }
    }
    return res;
}

---------------------------------------------------------------------

//130 Surrounded Regions 包围区域
Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example:
    X X X X
    X O O X
    X X O X
    X O X X
After running your function, the board should be:
    X X X X
    X X X X
    X X X X
    X O X X
Explanation:
    Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.


void solve(vector<vector<char> >& board) {
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[i].size(); ++j) {
            if ((i == 0 || i == board.size() - 1 || j == 0 || j == board[i].size() - 1) && board[i][j] == 'O')
                solveDFS(board, i, j);
        }
    }
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[i].size(); ++j) {
            if (board[i][j] == 'O') board[i][j] = 'X';
            if (board[i][j] == '$') board[i][j] = 'O';
        }
    }
}

void solveDFS(vector<vector<char> > &board, int i, int j) {
    if (board[i][j] == 'O') {
        board[i][j] = '$';
        if (i > 0 && board[i - 1][j] == 'O') 
            solveDFS(board, i - 1, j);
        if (j < board[i].size() - 1 && board[i][j + 1] == 'O') 
            solveDFS(board, i, j + 1);
        if (i < board.size() - 1 && board[i + 1][j] == 'O') 
            solveDFS(board, i + 1, j);
        if (j > 0 && board[i][j - 1] == 'O') 
            solveDFS(board, i, j - 1);
    }
}

---------------------------------------------------------------------

//131 Palindrome Partitioning 分割回文串
Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

Example:
Input: "aab"
Output:
    [
      ["aa","b"],
      ["a","a","b"]
    ]

深度优先搜索
由于不知道该如何切割，所以我们要遍历所有的切割情况，即一个字符，两个字符，三个字符，等等
vector<vector<string>> partition(string s) {
    vector<vector<string>> res;
    vector<string> out;
    helper(s, 0, out, res);
    return res;
}

void helper(string s, int start, vector<string>& out, vector<vector<string>>& res) {
    if (start == s.size()) { res.push_back(out); return; }
    for (int i = start; i < s.size(); ++i) {
        if (!isPalindrome(s, start, i)) continue;
        out.push_back(s.substr(start, i - start + 1));
        helper(s, i + 1, out, res);
        out.pop_back();
    }
}

bool isPalindrome(string s, int start, int end) {
    while (start < end) {
        if (s[start] != s[end]) return false;
        ++start; --end;
    }
    return true;
}

---------------------------------------------------------------------

//133 Clone Graph 克隆无向图
Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph. Each node in the graph contains a val (int) and a list (List[Node]) of its neighbors.

Example:
    Input:
    {"$id":"1","neighbors":[{"$id":"2","neighbors":[{"$ref":"1"},{"$id":"3","neighbors":[{"$ref":"2"},{"$id":"4","neighbors":[{"$ref":"3"},{"$ref":"1"}],"val":4}],"val":3}],"val":2},{"$ref":"4"}],"val":1}

Explanation:
    Node 1's value is 1, and it has two neighbors: Node 2 and 4.
    Node 2's value is 2, and it has two neighbors: Node 1 and 3.
    Node 3's value is 3, and it has two neighbors: Node 2 and 4.
    Node 4's value is 4, and it has two neighbors: Node 1 and 3.
 
Note:
The number of nodes will be between 1 and 100.
The undirected graph is a simple graph, which means no repeated edges and no self-loops in the graph.
Since the graph is undirected, if node p has node q as neighbor, then node q must have node p as neighbor too.
You must return the copy of the given node as a reference to the cloned graph.

对于图的遍历的两大基本方法是深度优先搜索 DFS 和广度优先搜索 BFS

DFS
Node* cloneGraph(Node* node) {
    unordered_map<Node*, Node*> m;
    return helper(node, m);
}

Node* helper(Node* node, unordered_map<Node*, Node*>& m) {
    if (!node) return NULL;
    if (m.count(node)) return m[node];
    Node *clone = new Node(node->val);
    m[node] = clone;
    for (Node *neighbor : node->neighbors) {
        clone->neighbors.push_back(helper(neighbor, m));
    }
    return clone;
}

BFS
Node* cloneGraph(Node* node) {
    if (!node) return NULL;
    unordered_map<Node*, Node*> m;
    queue<Node*> q{{node}};
    Node *clone = new Node(node->val);
    m[node] = clone;
    while (!q.empty()) {
        Node *t = q.front(); q.pop();
        for (Node *neighbor : t->neighbors) {
            if (!m.count(neighbor)) {
                m[neighbor] = new Node(neighbor->val);
                q.push(neighbor);
            }
            m[t]->neighbors.push_back(m[neighbor]);
        }
    }
    return clone;
}

---------------------------------------------------------------------

//134 Gas Station 加油站问题
There are N gas stations along a circular route, where the amount of gas at station i is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.

Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.

Note:
If there exists a solution, it is guaranteed to be unique.
Both input arrays are non-empty and have the same length.
Each element in the input arrays is a non-negative integer.

Example 1:
    Input: 
    gas  = [1,2,3,4,5]
    cost = [3,4,5,1,2]

    Output: 3

Explanation:
    Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    Travel to station 4. Your tank = 4 - 1 + 5 = 8
    Travel to station 0. Your tank = 8 - 2 + 1 = 7
    Travel to station 1. Your tank = 7 - 3 + 2 = 6
    Travel to station 2. Your tank = 6 - 4 + 3 = 5
    Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
    Therefore, return 3 as the starting index.
    
Example 2:
    Input: 
    gas  = [2,3,4]
    cost = [3,4,3]

    Output: -1
    
Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.

int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int total = 0, sum = 0, start = 0;
    for (int i = 0; i < gas.size(); ++i) {
        total += gas[i] - cost[i];
        sum += gas[i] - cost[i];
        if (sum < 0) {
            start = i + 1;
            sum = 0;
        }
    }
    return (total < 0) ? -1 : start;
}

---------------------------------------------------------------------

//137 Single Number II 单独的数字之二
Given a non-empty array of integers, every element appears three times except for one, which appears exactly once. Find that single one.

Note:

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

Example 1:

Input: [2,2,3,2]
Output: 3
Example 2:

Input: [0,1,0,1,0,1,99]
Output: 99

如果某一位上为1的话，那么如果该整数出现了三次，对3取余为0，最终剩下来的那个数就是单独的数字

int singleNumber(vector<int>& nums) {
    int res = 0;
    for (int i = 0; i < 32; ++i) {
        int sum = 0;
        for (int j = 0; j < nums.size(); ++j) {
            sum += (nums[j] >> i) & 1;
        }
        res |= (sum % 3) << i;
    }
    return res;
}

---------------------------------------------------------------------

//138 Copy List with Random Pointer 拷贝带有随机指针的链表
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

Example 1:
    Input:
    {"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.

Node* copyRandomList(Node* head) {
    if (!head) return nullptr;
    Node *res = new Node(head->val, nullptr, nullptr);
    Node *node = res, *cur = head->next;
    unordered_map<Node*, Node*> m;
    m[head] = res;
    while (cur) {
        Node *t = new Node(cur->val, nullptr, nullptr);
        node->next = t;
        m[cur] = t;
        node = node->next;
        cur = cur->next;
    }
    node = res; cur = head;
    while (cur) {
        node->random = m[cur->random];
        node = node->next;
        cur = cur->next;
    }
    return res;
}

Node* copyRandomList(Node* head) {
    unordered_map<Node*, Node*> m;
    return helper(head, m);
}

Node* helper(Node* node, unordered_map<Node*, Node*>& m) {
    if (!node) return nullptr;
    if (m.count(node)) return m[node];
    Node *res = new Node(node->val, nullptr, nullptr);
    m[node] = res;
    res->next = helper(node->next, m);
    res->random = helper(node->random, m);
    return res;
}

---------------------------------------------------------------------

//139 Word Break 拆分词句
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Note:
The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.

Example 1:
    Input: s = "leetcode", wordDict = ["leet", "code"]
    Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Example 2:
    Input: s = "applepenapple", wordDict = ["apple", "pen"]
    Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.

Example 3:
    Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
    Output: false

子数组或者子字符串且求极值的题，基本就是DP


bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
    vector<bool> dp(s.size() + 1);
    dp[0] = true;
    for (int i = 0; i < dp.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (dp[j] && wordSet.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp.back();
}
    
---------------------------------------------------------------------

//142 Linked List Cycle II 单链表中的环之二
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Note: Do not modify the linked list.

Example 1:
    Input: head = [3,2,0,-4], pos = 1
    Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.

Example 2:
    Input: head = [1,2], pos = 0
    Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the first node.

Example 3:
    Input: head = [1], pos = -1
    Output: no cycle
Explanation: There is no cycle in the linked list. 

Follow-up:
Can you solve it without using extra space?

ListNode *detectCycle(ListNode *head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) break;
    }
    if (!fast || !fast->next) return NULL;
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    return fast;
}

---------------------------------------------------------------------

//143 Reorder List 重排链表
Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example 1:
    Given 1->2->3->4, reorder it to 1->4->2->3.
Example 2:
    Given 1->2->3->4->5, reorder it to 1->5->2->4->3.

使用快慢指针来找到链表的中点，并将链表从中点处断开，形成两个独立的链表。
将第二个链翻转
将第二个链表的元素间隔地插入第一个链表中
    
void reorderList(ListNode *head) {
    if (!head || !head->next || !head->next->next) return;
    ListNode *fast = head, *slow = head;
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    ListNode *mid = slow->next;
    slow->next = NULL;
    ListNode *last = mid, *pre = NULL;
    while (last) {
        ListNode *next = last->next;
        last->next = pre;
        pre = last;
        last = next;
    }
    while (head && pre) {
        ListNode *next = head->next;
        head->next = pre;
        pre = pre->next;
        head->next->next = next;
        head = next;
    }
}

---------------------------------------------------------------------

//144 Binary Tree Preorder Traversal 二叉树的先序遍历
Given a binary tree, return the preorder traversal of its nodes' values.

Example:
    Input: [1,null,2,3]
       1
        \
         2
        /
       3

    Output: [1,2,3]
Follow up: Recursive solution is trivial, could you do it iteratively?


把根节点push到栈中
循环检测栈是否为空，若不空，则取出栈顶元素，保存其值，然后看其右子节点是否存在，若存在则push到栈中。再看其左子节点，若存在，则push到栈中

vector<int> preorderTraversal(TreeNode* root) {
    if (!root) return {};
    vector<int> res;
    stack<TreeNode*> s{{root}};
    while (!s.empty()) {
        TreeNode *t = s.top(); s.pop();
        res.push_back(t->val);
        if (t->right) s.push(t->right);
        if (t->left) s.push(t->left);
    }
    return res;
}

---------------------------------------------------------------------

//146 LRU Cache 最近最少使用页面置换缓存器
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a positive capacity.

Follow up:
Could you do both operations in O(1) time complexity?

Example:
    LRUCache cache = new LRUCache( 2 /* capacity */ );

    cache.put(1, 1);
    cache.put(2, 2);
    cache.get(1);       // returns 1
    cache.put(3, 3);    // evicts key 2
    cache.get(2);       // returns -1 (not found)
    cache.put(4, 4);    // evicts key 1
    cache.get(1);       // returns -1 (not found)
    cache.get(3);       // returns 3
    cache.get(4);       // returns 4

class LRUCache{
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        auto it = m.find(key);
        if (it == m.end()) return -1;
        l.splice(l.begin(), l, it->second);
        return it->second->second;
    }
    
    void put(int key, int value) {
        auto it = m.find(key);
        if (it != m.end()) l.erase(it->second);
        l.push_front(make_pair(key, value));
        m[key] = l.begin();
        if (m.size() > cap) {
            int k = l.rbegin()->first;
            l.pop_back();
            m.erase(k);
        }
    }
    
private:
    int cap;
    list<pair<int, int>> l;
    unordered_map<int, list<pair<int, int>>::iterator> m;
};

---------------------------------------------------------------------

//147 Insertion Sort List 链表插入排序
Sort a linked list using insertion sort.

A graphical example of insertion sort. The partial sorted list (black) initially contains only the first element in the list.
With each iteration one element (red) is removed from the input data and inserted in-place into the sorted list
 
Algorithm of Insertion Sort:

Insertion sort iterates, consuming one input element each repetition, and growing a sorted output list.
At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list, and inserts it there.
It repeats until no input elements remain.

Example 1:
    Input: 4->2->1->3
    Output: 1->2->3->4

Example 2:
    Input: -1->5->3->4->0
    Output: -1->0->3->4->5

ListNode* insertionSortList(ListNode* head) {
    ListNode *dummy = new ListNode(-1), *cur = dummy;
    while (head) {
        ListNode *t = head->next;
        cur = dummy;
        while (cur->next && cur->next->val <= head->val) {
            cur = cur->next;
        }
        head->next = cur->next;
        cur->next = head;
        head = t;
    }
    return dummy->next;
}

---------------------------------------------------------------------

//148 Sort List 链表排序
Sort a linked list in O(n log n) time using constant space complexity.

Example 1:
    Input: 4->2->1->3
    Output: 1->2->3->4
    
Example 2:
    Input: -1->5->3->4->0
    Output: -1->0->3->4->5

ListNode* sortList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode *slow = head, *fast = head, *pre = head;
    while (fast && fast->next) {
        pre = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    pre->next = NULL;
    return merge(sortList(head), sortList(slow));
}
ListNode* merge(ListNode* l1, ListNode* l2) {
    ListNode *dummy = new ListNode(-1);
    ListNode *cur = dummy;
    while (l1 && l2) {
        if (l1->val < l2->val) {
            cur->next = l1;
            l1 = l1->next;
        } else {
            cur->next = l2;
            l2 = l2->next;
        }
        cur = cur->next;
    }
    if (l1) cur->next = l1;
    if (l2) cur->next = l2;
    return dummy->next;
}

---------------------------------------------------------------------

//150 Evaluate Reverse Polish Notation 计算逆波兰表达式
Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are +, -, *, /. Each operand may be an integer or another expression.

Note:

Division between two integers should truncate toward zero.
The given RPN expression is always valid. That means the expression would always evaluate to a result and there won't be any divide by zero operation.

Example 1:
    Input: ["2", "1", "+", "3", "*"]
    Output: 9

Explanation: ((2 + 1) * 3) = 9

Example 2:
    Input: ["4", "13", "5", "/", "+"]
    Output: 6

Explanation: (4 + (13 / 5)) = 6

Example 3:
    Input: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    Output: 22

Explanation: 
      ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
    = ((10 * (6 / (12 * -11))) + 17) + 5
    = ((10 * (6 / -132)) + 17) + 5
    = ((10 * 0) + 17) + 5
    = (0 + 17) + 5
    = 17 + 5
    = 22

int evalRPN(vector<string>& tokens) {
    if (tokens.size() == 1) return stoi(tokens[0]);
    stack<int> st;
    for (int i = 0; i < tokens.size(); ++i) {
        if (tokens[i] != "+" && tokens[i] != "-" && tokens[i] != "*" && tokens[i] != "/") {
            st.push(stoi(tokens[i]));
        } else {
            int num1 = st.top(); st.pop();
            int num2 = st.top(); st.pop();
            if (tokens[i] == "+") st.push(num2 + num1);
            if (tokens[i] == "-") st.push(num2 - num1);
            if (tokens[i] == "*") st.push(num2 * num1);
            if (tokens[i] == "/") st.push(num2 / num1);
        }
    }
    return st.top();
}

---------------------------------------------------------------------

//151 Reverse Words in a String 翻转字符串中的单词
Given an input string, reverse the string word by word.

Example 1:
    Input: "the sky is blue"
    Output: "blue is sky the"
    
Example 2:
    Input: "  hello world!  "
    Output: "world! hello"

Explanation: Your reversed string should not contain leading or trailing spaces.

Example 3:
    Input: "a good   example"
    Output: "example good a"
    
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.

Note:
A word is defined as a sequence of non-space characters.
Input string may contain leading or trailing spaces. However, your reversed string should not contain leading or trailing spaces.
You need to reduce multiple spaces between two words to a single space in the reversed string.
 
Follow up:

For C programmers, try to solve it in-place in O(1) extra space.

void reverseWords(string &s) {
    int storeIndex = 0, n = s.size();
    reverse(s.begin(), s.end());
    for (int i = 0; i < n; ++i) {
        if (s[i] != ' ') {
            if (storeIndex != 0) s[storeIndex++] = ' ';
            int j = i;
            while (j < n && s[j] != ' ') s[storeIndex++] = s[j++];
            reverse(s.begin() + storeIndex - (j - i), s.begin() + storeIndex);
            i = j;
        }
    }
    s.resize(storeIndex);
}

---------------------------------------------------------------------

//152 Maximum Product Subarray 求最大子数组乘积
Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

Example 1:
    Input: [2,3,-2,4]
    Output: 6

Explanation: [2,3] has the largest product 6.

Example 2:
    Input: [-2,0,-1]
    Output: 0

Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

DP
要维护两个dp数组，其中f[i]表示子数组[0, i]范围内并且一定包含nums[i]数字的最大子数组乘积，g[i]表示子数组[0, i]范围内并且一定包含nums[i]数字的最小子数组乘积，维护最小的是因为一旦遇到一个负数，最小的会成为最大的

int maxProduct(vector<int>& nums) {
    int res = nums[0], n = nums.size();
    vector<int> f(n, 0), g(n, 0);
    f[0] = nums[0];
    g[0] = nums[0];
    for (int i = 1; i < n; ++i) {
        f[i] = max(max(f[i - 1] * nums[i], g[i - 1] * nums[i]), nums[i]);
        g[i] = min(min(f[i - 1] * nums[i], g[i - 1] * nums[i]), nums[i]);
        res = max(res, f[i]);
    }
    return res;
}

可以优化空间，只用2个变量就行
int maxProduct(vector<int>& nums) {
    if (nums.empty()) return 0;
    int res = nums[0], mn = nums[0], mx = nums[0];
    for (int i = 1; i < nums.size(); ++i) {
        int tmax = mx, tmin = mn;
        mx = max(max(nums[i], tmax * nums[i]), tmin * nums[i]);
        mn = min(min(nums[i], tmax * nums[i]), tmin * nums[i]);
        res = max(res, mx);
    }
    return res;
}

---------------------------------------------------------------------

//153 Find Minimum in Rotated Sorted Array 寻找旋转有序数组的最小值
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.

Example 1:
    Input: [3,4,5,1,2] 
    Output: 1

Example 2:
    Input: [4,5,6,7,0,1,2]
    Output: 0

int findMin(vector<int>& nums) {
    int left = 0, right = (int)nums.size() - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[right]) left = mid + 1;
        else right = mid;
    }
    return nums[right];
}

---------------------------------------------------------------------

//156 Binary Tree Upside Down 上下翻转二叉树
Given a binary tree where all the right nodes are either leaf nodes with a sibling (a left node that shares the same parent node) or empty, flip it upside down and turn it into a tree where the original right nodes turned into left leaf nodes. Return the new root.

For example:

Given a binary tree {1,2,3,4,5},
        1
       / \
      2   3
     / \
    4   5

return the root of the binary tree [4,5,2,#,#,3,1].
       4
      / \
     5   2
        / \
       3   1  

二叉树上下颠倒一下，而且限制了右节点要么为空要么一定会有对应的左节点。上下颠倒后原来二叉树的最左子节点变成了根节点，其对应的右节点变成了其左子节点，其父节点变成了其右子节点，相当于顺时针旋转了一下

TreeNode *upsideDownBinaryTree(TreeNode *root) {
    TreeNode *cur = root, *pre = NULL, *next = NULL, *tmp = NULL;
    while (cur) {
        next = cur->left;
        cur->left = tmp;
        tmp = cur->right;
        cur->right = pre;
        pre = cur;
        cur = next;
    }
    return pre;
}

---------------------------------------------------------------------

//161 One Edit Distance 一个编辑距离
Given two strings s and t, determine if they are both one edit distance apart.

Note: 
There are 3 possiblities to satisify one edit distance apart:

Insert a character into s to get t
Delete a character from s to get t
Replace a character of s to get t
Example 1:

Input: s = "ab", t = "acb"
Output: true
Explanation: We can insert 'c' into s to get t.
Example 2:

Input: s = "cab", t = "ad"
Output: false
Explanation: We cannot get t from s by only one step.
Example 3:

Input: s = "1203", t = "1213"
Output: true
Explanation: We can replace '0' with '1' to get t.

bool isOneEditDistance(string s, string t) {
    if (s.size() < t.size()) swap(s, t);
    int m = s.size(), n = t.size(), diff = m - n;
    if (diff >= 2) return false;
    else if (diff == 1) {
        for (int i = 0; i < n; ++i) {
            if (s[i] != t[i]) {
                return s.substr(i + 1) == t.substr(i);
            }
        }
        return true;
    } else {
        int cnt = 0;
        for (int i = 0; i < m; ++i) {
            if (s[i] != t[i]) ++cnt;
        }
        return cnt == 1;
    }
}

---------------------------------------------------------------------

//162 Find Peak Element 求数组的局部峰值
A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

Example 1:
    Input: nums = [1,2,3,1]
    Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Example 2:
    Input: nums = [1,2,1,3,5,6,4]
    Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.

Note:
Your solution should be in logarithmic complexity.

int findPeakElement(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < nums[mid + 1]) left = mid + 1;
        else right = mid;
    }
    return right;
}

---------------------------------------------------------------------

//163 Missing Ranges 缺失区间
Given a sorted integer array where the range of elements are [0, 99] inclusive, return its missing ranges.
For example, given [0, 1, 3, 50, 75], return [“2”, “4->49”, “51->74”, “76->99”]

vector<string> findMissingRanges(vector<int>& nums, int lower, int upper) {
    vector<string> res;
    int l = lower;
    for (int i = 0; i <= nums.size(); ++i) {
        int r = (i < nums.size() && nums[i] <= upper) ? nums[i] : upper + 1;
        if (l == r) ++l;
        else if (r > l) {
            res.push_back(r - l == 1 ? to_string(l) : to_string(l) + "->" + to_string(r - 1));
            l = r + 1;
        }
    }
    return res;
}

---------------------------------------------------------------------

//165 Compare Version Numbers 比较版本号
Compare two version numbers version1 and version2.
If version1 > version2 return 1; if version1 < version2 return -1;otherwise return 0.

You may assume that the version strings are non-empty and contain only digits and the . character.

The . character does not represent a decimal point and is used to separate number sequences.

For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.

You may assume the default revision number for each level of a version number to be 0. For example, version number 3.4 has a revision number of 3 and 4 for its first and second level revision number. Its third and fourth level revision number are both 0.

Example 1:
    Input: version1 = "0.1", version2 = "1.1"
    Output: -1

Example 2:
    Input: version1 = "1.0.1", version2 = "1"
    Output: 1

Example 3:
    Input: version1 = "7.5.2.4", version2 = "7.5.3"
    Output: -1

Example 4:
    Input: version1 = "1.01", version2 = "1.001"
    Output: 0
Explanation: Ignoring leading zeroes, both “01” and “001" represent the same number “1”

Example 5:
    Input: version1 = "1.0", version2 = "1.0.0"
    Output: 0
Explanation: The first version number does not have a third level revision number, which means its third level revision number is default to "0"

Note:
    Version strings are composed of numeric strings separated by dots . and this numeric strings may have leading zeroes.
    Version strings do not start or end with dots, and they will not be two consecutive dots.

int compareVersion(string version1, string version2) {
    int n1 = version1.size(), n2 = version2.size();
    int i = 0, j = 0, d1 = 0, d2 = 0;
    while (i < n1 || j < n2) {
        while (i < n1 && version1[i] != '.') {
            d1 = d1 * 10 + version1[i++] - '0';
        }
        while (j < n2 && version2[j] != '.') {
            d2 = d2 * 10 + version2[j++] - '0';
        }
        if (d1 > d2) return 1;
        else if (d1 < d2) return -1;
        d1 = d2 = 0;
        ++i; ++j;
    }
    return 0;
}

---------------------------------------------------------------------

//166 Fraction to Recurring Decimal 分数转循环小数
Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

Example 1:
    Input: numerator = 1, denominator = 2
    Output: "0.5"

Example 2:
    Input: numerator = 2, denominator = 1
    Output: "2"

Example 3:
    Input: numerator = 2, denominator = 3
    Output: "0.(6)"

string fractionToDecimal(int numerator, int denominator) {
    int s1 = numerator >= 0 ? 1 : -1;
    int s2 = denominator >= 0 ? 1 : -1;
    long long num = abs( (long long)numerator );
    long long den = abs( (long long)denominator );
    long long out = num / den;
    long long rem = num % den;
    unordered_map<long long, int> m;
    string res = to_string(out);
    if (s1 * s2 == -1 && (out > 0 || rem > 0)) res = "-" + res;
    if (rem == 0) return res;
    res += ".";
    string s = "";
    int pos = 0;
    while (rem != 0) {
        if (m.find(rem) != m.end()) {
            s.insert(m[rem], "(");
            s += ")";
            return res + s;
        }
        m[rem] = pos;
        s += to_string((rem * 10) / den);
        rem = (rem * 10) % den;
        ++pos;
    }
    return res + s;
}

---------------------------------------------------------------------

//173 Binary Search Tree Iterator 二叉搜索树迭代器
Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

Example:
    BSTIterator iterator = new BSTIterator(root);
    iterator.next();    // return 3
    iterator.next();    // return 7
    iterator.hasNext(); // return true
    iterator.next();    // return 9
    iterator.hasNext(); // return true
    iterator.next();    // return 15
    iterator.hasNext(); // return true
    iterator.next();    // return 20
    iterator.hasNext(); // return false 

Note:
next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
You may assume that next() call will always be valid, that is, there will be at least a next smallest number in the BST when next() is called.

class BSTIterator {
public:
    BSTIterator(TreeNode *root) {
        while (root) {
            s.push(root);
            root = root->left;
        }
    }

    bool hasNext() {
        return !s.empty();
    }

    int next() {
        TreeNode *n = s.top();
        s.pop();
        int res = n->val;
        if (n->right) {
            n = n->right;
            while (n) {
                s.push(n);
                n = n->left;
            }
        }
        return res;
    }
private:
    stack<TreeNode*> s;
};

---------------------------------------------------------------------

//179 Largest Number 最大数
Given a list of non negative integers, arrange them such that they form the largest number.

Example 1:
    Input: [10,2]
    Output: "210"

Example 2:
    Input: [3,30,34,5,9]
    Output: "9534330"

降序
bool compare(int a, int b) {
    return to_string(a) + to_string(b) > to_string(b) + to_string(a); 
}

string largestNumber(vector<int>& nums) {
    string res;
    sort(nums.begin(), nums.end(), compare);
    for (int i = 0; i < nums.size(); ++i) {
        res += to_string(nums[i]);
    }
    return res[0] == '0' ? "0" : res;
}

---------------------------------------------------------------------

//186 Reverse Words in a String II 翻转字符串中的单词之二
Given an input string , reverse the string word by word.

Example:
    Input:  ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
    Output: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]

Note: 
    A word is defined as a sequence of non-space characters.
    The input string does not contain leading or trailing spaces.
    The words are always separated by a single space.

Follow up: Could you do it in-place without allocating extra space?

void reverseWords(vector<char>& str) {
    int left = 0, n = str.size();
    for (int i = 0; i <= n; ++i) {
        if (i == n || str[i] == ' ') {
            reverse(str, left, i - 1);
            left = i + 1;
        }
    }
    reverse(str, 0, n - 1);
}

void reverse(vector<char>& str, int left, int right) {
    while (left < right) {
        char t = str[left];
        str[left] = str[right];
        str[right] = t;
        ++left; --right;
    }
}

---------------------------------------------------------------------

//187 Repeated DNA Sequences 求重复的DNA序列
All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When studying DNA, it is sometimes useful to identify repeated sequences within the DNA.

Write a function to find all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule.

Example:
    Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
    Output: ["AAAAACCCCC", "CCCCCAAAAA"]

vector<string> findRepeatedDnaSequences(string s) {
    unordered_set<string> res, st;
    for (int i = 0; i + 9 < s.size(); ++i) {
        string t = s.substr(i, 10);
        if (st.count(t)) res.insert(t);
        else st.insert(t);
    }
    return vector<string>{res.begin(), res.end()};
}

为了节省空间，将string缩小为int，因为2位bit就能区分A C G T四个字符，十个字母也就20位，所以可以用int代替
vector<string> findRepeatedDnaSequences(string s) {
    vector<string> res;
    unordered_map<int, int> st;  //存储出现过的十个字符
    unordered_map<int, int> m{{'A', 0}, {'C', 1}, {'G', 2}, {'T', 3}};
    int cur = 0;
    for (int i = 0; i < 9; ++i) cur = cur << 2 | m[s[i]];
    for (int i = 9; i < s.size(); ++i) {
        cur = ((cur & 0x3ffff) << 2) | (m[s[i]]);
        if (st.count(cur)) {
            if (st[cur] == 1) {
                res.push_back(s.substr(i - 9, 10));
            }
            ++st[cur];
        }
        else st[cur] = 1;
    }
    return res;
}
    
---------------------------------------------------------------------

//199 Binary Tree Right Side View 二叉树的右侧视图
Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Example:
    Input: [1,2,3,null,5,null,4]
    Output: [1, 3, 4]
    
Explanation:
       1            <---
     /   \
    2     3         <---
     \     \
      5     4       <---

vector<int> rightSideView(TreeNode *root) {
    vector<int> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        res.push_back(q.back()->val);
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            TreeNode *node = q.front();
            q.pop();
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return res;
}

---------------------------------------------------------------------

//200 Number of Islands 岛屿的数量
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:
    Input:
        11110
        11010
        11000
        00000
    Output: 1

Example 2:
    Input:
        11000
        11000
        00100
        00011
    Output: 3

DFS
int numIslands(vector<vector<char>>& grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int m = grid.size(), n = grid[0].size(), res = 0;
    vector<vector<bool>> visited(m, vector<bool>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == '0' || visited[i][j]) continue;
            helper(grid, visited, i, j);
            ++res;
        }
    }
    return res;
}

void helper(vector<vector<char>>& grid, vector<vector<bool>>& visited, int x, int y) {
    if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == '0' || visited[x][y]) return;
    visited[x][y] = true;
    helper(grid, visited, x - 1, y);
    helper(grid, visited, x + 1, y);
    helper(grid, visited, x, y - 1);
    helper(grid, visited, x, y + 1);
}

递归也可以用队列来替代
int numIslands(vector<vector<char>>& grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int m = grid.size(), n = grid[0].size(), res = 0;
    vector<vector<bool>> visited(m, vector<bool>(n));
    vector<int> dirX{-1, 0, 1, 0}, dirY{0, 1, 0, -1};
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == '0' || visited[i][j]) continue;
            ++res;
            queue<int> q{{i * n + j}};
            while (!q.empty()) {
                int t = q.front(); q.pop();
                for (int k = 0; k < 4; ++k) {
                    int x = t / n + dirX[k], y = t % n + dirY[k];
                    if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == '0' || visited[x][y]) continue;
                    visited[x][y] = true;
                    q.push(x * n + y);
                }
            }
        }
    }
    return res;
}

---------------------------------------------------------------------
