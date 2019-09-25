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
There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example 1:
	Input: 2, [[1,0]] 
	Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished   
             course 0. So the correct course order is [0,1] .

Example 2:
	Input: 4, [[1,0],[2,0],[3,1],[3,2]]
	Output: [0,1,2,3] or [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both     
             courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. 
             So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3] .

Note:
The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.

和课程表的题类似，只是从queue中每取出一个数组就将其存在结果中
vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
	vector<int> res;
	vector<vector<int> > graph(numCourses, vector<int>(0));
	vector<int> in(numCourses, 0);
	for (auto &a : prerequisites) {
		graph[a.second].push_back(a.first);
		++in[a.first];
	}
	queue<int> q;
	for (int i = 0; i < numCourses; ++i) {
		if (in[i] == 0) q.push(i);
	}
	while (!q.empty()) {
		int t = q.front();
		res.push_back(t);
		q.pop();
		for (auto &a : graph[t]) {
			--in[a];
			if (in[a] == 0) q.push(a);
		}
	}
	if (res.size() != numCourses) res.clear();
	return res;
}
	
---------------------------------------------------------------------

//211 Add and Search Word - Data structure design 添加与搜索单词 - 数据结构设计
Design a data structure that supports the following two operations:
void addWord(word)
bool search(word)
search(word) can search a literal word or a regular expression string containing only letters a-z or .. A. means it can represent any one letter.

For example:
	addWord("bad")
	addWord("dad")
	addWord("mad")
	search("pad") -> false
	search("bad") -> true
	search(".ad") -> true
	search("b..") -> true

Note:
You may assume that all words are consist of lowercase letters a-z.

click to show hint.

You should be familiar with how a Trie works. If not, please work on this problem: Implement Trie (Prefix Tree) first.

和之前的字典树类似
class WordDictionary {
public:
    struct TrieNode {
    public:
        TrieNode *child[26];
        bool isWord;
        TrieNode() : isWord(false) {
            for (auto &a : child) a = NULL;
        }
    };
    
    WordDictionary() {
        root = new TrieNode();
    }
    
    // Adds a word into the data structure.
    void addWord(string word) {
        TrieNode *p = root;
        for (auto &a : word) {
            int i = a - 'a';
            if (!p->child[i]) p->child[i] = new TrieNode();
            p = p->child[i];
        }
        p->isWord = true;
    }

    // Returns if the word is in the data structure. A word could
    // contain the dot character '.' to represent any one letter.
    bool search(string word) {
        return searchWord(word, root, 0);
    }
    
    bool searchWord(string &word, TrieNode *p, int i) {
        if (i == word.size()) return p->isWord;
        if (word[i] == '.') {
            for (auto &a : p->child) {
                if (a && searchWord(word, a, i + 1)) return true;
            }
            return false;
        } else {
            return p->child[word[i] - 'a'] && searchWord(word, p->child[word[i] - 'a'], i + 1);
        }
    }
    
private:
    TrieNode *root;
};

---------------------------------------------------------------------

//213 House Robber II 打家劫舍II
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:

Input: [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2),
             because they are adjacent houses.
Example 2:

Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
Credits:
Special thanks to @Freezen for adding this problem and creating all test cases.

int rob(vector<int>& nums) {
	if (nums.size() <= 1) return nums.empty() ? 0 : nums[0];
	return max(rob(nums, 0, nums.size() - 1), rob(nums, 1, nums.size()));
}

int rob(vector<int> &nums, int left, int right) {
	if (right - left <= 1) return nums[left];
	vector<int> dp(right, 0);
	dp[left] = nums[left];
	dp[left + 1] = max(nums[left], nums[left + 1]);
	for (int i = left + 2; i < right; ++i) {
		dp[i] = max(nums[i] + dp[i - 2], dp[i - 1]);
	}
	return dp.back();
}
	
可以使用两个变量来代替整个DP数组
robEven就是要抢偶数位置的房子，robOdd就是要抢奇数位置的房子
int rob(vector<int>& nums) {
	if (nums.size() <= 1) return nums.empty() ? 0 : nums[0];
	return max(rob(nums, 0, nums.size() - 1), rob(nums, 1, nums.size()));
}

int rob(vector<int> &nums, int left, int right) {
	int robEven = 0, robOdd = 0;
	for (int i = left; i < right; ++i) {
		if (i % 2 == 0) {
			robEven = max(robEven + nums[i], robOdd);
		} else {
			robOdd = max(robEven, robOdd + nums[i]);
		}
	}
	return max(robEven, robOdd);
}

---------------------------------------------------------------------

//215 Kth Largest Element in an Array 数组中第k大的元素
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:
	Input: [3,2,1,5,6,4] and k = 2
	Output: 5
	
Example 2:
	Input: [3,2,3,1,2,4,5,5,6] and k = 4
	Output: 4
	
Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.

先排序
int findKthLargest(vector<int>& nums, int k) {
	sort(nums.begin(), nums.end());
	return nums[nums.size() - k];
}

快速排序思想
int findKthLargest(vector<int>& nums, int k) {
	int left = 0, right = nums.size() - 1;
	while (true) {
		int pos = partition(nums, left, right);
		if (pos == k - 1) return nums[pos];
		if (pos > k - 1) right = pos - 1;
		else left = pos + 1;
	}
}

int partition(vector<int>& nums, int left, int right) {
	int pivot = nums[left], l = left + 1, r = right;
	while (l <= r) {
		if (nums[l] < pivot && nums[r] > pivot) {
			swap(nums[l++], nums[r--]);
		}
		if (nums[l] >= pivot) ++l;
		if (nums[r] <= pivot) --r;
	}
	swap(nums[left], nums[r]);
	return r;
}

---------------------------------------------------------------------

//216 Combination Sum III 组合之和之三
Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Ensure that numbers within the set are sorted in ascending order.

Example 1:
	Input: k = 3, n = 7
	Output:	[[1,2,4]]
	
Example 2:
	Input: k = 3, n = 9
	Output:	[[1,2,6], [1,3,5], [2,3,4]]
	
Credits:
Special thanks to @mithmatt for adding this problem and creating all test cases.

vector<vector<int> > combinationSum3(int k, int n) {
	vector<vector<int> > res;
	vector<int> out;
	combinationSum3DFS(k, n, 1, out, res);
	return res;
}

void combinationSum3DFS(int k, int n, int level, vector<int> &out, vector<vector<int> > &res) {
	if (n < 0) return;
	if (n == 0 && out.size() == k) res.push_back(out);
	for (int i = level; i <= 9; ++i) {
		out.push_back(i);
		combinationSum3DFS(k, n - i, i + 1, out, res);
		out.pop_back();
	}
}
	
---------------------------------------------------------------------

//220 Contains Duplicate III 包含重复值之三
Given an array of integers, find out whether there are two distinct indices i and j in the array such that the difference between nums[i] and nums[j] is at most t and the difference between i and j is at most k.

bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
	map<long long, int> m;
	int j = 0;
	for (int i = 0; i < nums.size(); ++i) {
		if (i - j > k) m.erase(nums[j++]);
		auto a = m.lower_bound((long long)nums[i] - t);
		if (a != m.end() && abs(a->first - nums[i]) <= t) return true;
		m[nums[i]] = i;
	}
	return false;
}

---------------------------------------------------------------------

//221 Maximal Square 最大正方形
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing all 1's and return its area.

For example, given the following matrix:
	1 0 1 0 0
	1 0 1 1 1
	1 1 1 1 1
	1 0 0 1 0
	Return 4.


---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------