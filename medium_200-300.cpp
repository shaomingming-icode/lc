#include "medium.h"

---------------------------------------------------------------------

//201 Bitwise AND of Numbers Range 数字范围位按位与
Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.

Example 1:
	Input: [5,7]
	Output: 4
Example 2:

Input: [0,1]
Output: 0

所有的数的左边共同的部分，也是m和n左边共同的部分
int rangeBitwiseAnd(int m, int n) {
	int d = INT_MAX;
	while ((m & d) != (n & d)) {
		d <<= 1;
	}
	return m & d;
}

---------------------------------------------------------------------

//207 Course Schedule 课程表
There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Example 1:

Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
Note:

The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.

LeetCode中关于图的题很少，有向图的仅此一道，还有一道关于无向图的题是 Clone Graph

BFS
思路是没人指向自己，自己只指向别人的线去掉，去完后如果还有线，那就说明有环
graph来表示有向图，数组 in 来表示每个顶点的入度
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
	vector<vector<int>> graph(numCourses, vector<int>());
	vector<int> in(numCourses);
	for (auto a : prerequisites) {
		graph[a[1]].push_back(a[0]);
		++in[a[0]];
	}
	queue<int> q;
	for (int i = 0; i < numCourses; ++i) {
		if (in[i] == 0) q.push(i);
	}
	while (!q.empty()) {
		int t = q.front(); q.pop();
		for (auto a : graph[t]) {
			--in[a];
			if (in[a] == 0) q.push(a);
		}
	}
	for (int i = 0; i < numCourses; ++i) {
		if (in[i] != 0) return false;
	}
	return true;
}

DFS
思路是从第一个门课开始，标记为访问，然后对新得到的课程递归，直到出现新的课程已经访问过了，则返回 false，否则true，然后把标记为已访问的课程改为未访问
graph来表示有向图，数组visit来记录访问状态，0表示还未访问过，1表示已经访问了，-1表示有冲突
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
	vector<vector<int>> graph(numCourses, vector<int>());
	vector<int> visit(numCourses);
	for (auto a : prerequisites) {
		graph[a[1]].push_back(a[0]);
	}
	for (int i = 0; i < numCourses; ++i) {
		if (!canFinishDFS(graph, visit, i)) return false;
	}
	return true;
}
bool canFinishDFS(vector<vector<int>>& graph, vector<int>& visit, int i) {
	if (visit[i] == -1) return false;
	if (visit[i] == 1) return true;
	visit[i] = -1;
	for (auto a : graph[i]) {
		if (!canFinishDFS(graph, visit, a)) return false;
	}
	visit[i] = 1;
	return true;
}

---------------------------------------------------------------------

//208 Implement Trie (Prefix Tree) 实现字典树(前缀树)
Implement a trie with insert, search, and startsWith methods.

Example:
	Trie trie = new Trie();
	trie.insert("apple");
	trie.search("apple");   // returns true
	trie.search("app");     // returns false
	trie.startsWith("app"); // returns true
	trie.insert("app");   
	trie.search("app");     // returns true
Note:
You may assume that all inputs are consist of lowercase letters a-z.
All inputs are guaranteed to be non-empty strings.

字典树主要有如下三点性质：
1. 根节点不包含字符，除根节点意外每个节点只包含一个字符。
2. 从根节点到某一个节点，路径上经过的字符连接起来，为该节点对应的字符串。
3. 每个节点的所有子节点包含的字符串不相同。

存储的方式有这三种
1. 对每个结点开一个字母集大小的数组，对应的下标是儿子所表示的字母，内容则是这个儿子对应在大数组上的位置，即标号
2. 对每个结点挂一个链表，按一定顺序记录每个儿子是谁
3. 使用左儿子右兄弟表示法记录这棵树
采用第一种方式

class TrieNode {
public:
    TrieNode *child[26];
    bool isWord;
    TrieNode(): isWord(false) {
        for (auto &a : child) a = nullptr;
    }
};

class Trie {
public:
    Trie() {
        root = new TrieNode();
    }
    void insert(string s) {
        TrieNode *p = root;
        for (auto &a : s) {
            int i = a - 'a';
            if (!p->child[i]) p->child[i] = new TrieNode();
            p = p->child[i];
        }
        p->isWord = true;
    }
    bool search(string key) {
        TrieNode *p = root;
        for (auto &a : key) {
            int i = a - 'a';
            if (!p->child[i]) return false;
            p = p->child[i];
        }
        return p->isWord;
    }
    bool startsWith(string prefix) {
        TrieNode *p = root;
        for (auto &a : prefix) {
            int i = a - 'a';
            if (!p->child[i]) return false;
            p = p->child[i];
        }
        return true;
    }
    
private:
    TrieNode* root;
};

---------------------------------------------------------------------

//209 Minimum Size Subarray Sum 长度最小的子数组
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

Example: 
	Input: s = 7, nums = [2,3,1,2,4,3]
	Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.

Follow up:
If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n). 

int minSubArrayLen(int s, vector<int>& nums) {
	int res = INT_MAX, left = 0, sum = 0;
	for (int i = 0; i < nums.size(); ++i) {
		sum += nums[i];
		while (left <= i && sum >= s) {
			res = min(res, i - left + 1);
			sum -= nums[left++];
		}
	}
	return res == INT_MAX ? 0 : res;
}

---------------------------------------------------------------------

//210 Course Schedule II 课程表II

---------------------------------------------------------------------



---------------------------------------------------------------------



---------------------------------------------------------------------



---------------------------------------------------------------------



---------------------------------------------------------------------






