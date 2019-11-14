#include "medium.h"

---------------------------------------------------------------------

320. Generalized Abbreviation
Write a function to generate the generalized abbreviations of a word. 
Note: The order of the output does not matter.

Example:
    Input: "word"
    Output:
    ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

vector<string> generateAbbreviations(string word) {
    vector<string> res{ word };
    helper(word, 0, res);
    return res;
}
void helper(string word, int pos, vector<string>& res) {
    for (int i = pos; i < word.size(); ++i) {  //从pos开始
        for (int j = 1; i + j <= word.size(); ++j) {  //数字
            string t = word.substr(0, i);
            t += to_string(j) + word.substr(i + j);
            res.push_back(t);
            helper(t, i + 1 + to_string(j).size(), res);
        }
    }
}

---------------------------------------------------------------------

322. Coin Change
You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

Example 1:
    Input: coins = [1, 2, 5], amount = 11
    Output: 3 
    Explanation: 11 = 5 + 5 + 1
Example 2:
    Input: coins = [2], amount = 3
    Output: -1
Note:
You may assume that you have an infinite number of each kind of coin.

int coinChange(vector<int>& coins, int amount) {
    vector<int> values(amount + 1, amount + 1);
    values[0] = 0;
    for (int i = 0; i <= amount; i++) {
        for (int j = 0; j < coins.size(); j++) {
            if (i - coins[j] >= 0) {
                values[i] = min(values[i - coins[j]] + 1, values[i]);
            }
        }
    }
    return values.back() > amount ? -1 : values.back();
}

---------------------------------------------------------------------

323. Number of Connected Components in an Undirected Graph
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.

Example 1:
    Input: n = 5 and edges = [[0, 1], [1, 2], [3, 4]]
     0          3
     |          |
     1 --- 2    4 
    Output: 2
Example 2:
    Input: n = 5 and edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
     0           4
     |           |
     1 --- 2 --- 3
    Output:  1
Note:
You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

int countComponents(int n, vector<vector<int>>& edges) {
    vector<vector<int>> graph(n);
    for (int i = 0; i < edges.size(); i++) {
        graph[edges[i][0]].push_back(edges[i][1]);
        graph[edges[i][1]].push_back(edges[i][0]);
    }

    vector<int> color(n, 0);
    int res = 0;
    for (int i = 0; i < n; i++) {
        if (color[i] != 0) {
            continue;
        }
        res++;
        color[i] = 1;
        queue<int> neighbor;
        neighbor.push(i);
        while (neighbor.size() != 0) {
            int node = neighbor.front();
            for (int j = 0; j < graph[node].size(); j++) {
                if (color[graph[node][j]] == 0) {
                    color[graph[node][j]] = 1;
                    neighbor.push(graph[node][j]);
                }
            }
            neighbor.pop();
        }
    }
    return res;
}

---------------------------------------------------------------------

324. Wiggle Sort II
Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....

Example 1:
    Input: nums = [1, 5, 1, 1, 6, 4]
    Output: One possible answer is [1, 4, 1, 5, 1, 6].
Example 2:
    Input: nums = [1, 3, 2, 2, 3, 1]
    Output: One possible answer is [2, 3, 1, 3, 1, 2].
Note:
You may assume all input has valid answer.

Follow Up:
Can you do it in O(n) time and/or in-place with O(1) extra space?

void wiggleSort(vector<int>& nums) {
    vector<int> temp = nums;
    sort(temp.begin(), temp.end());
    int small = (temp.size() + 1) / 2 - 1, big = temp.size() - 1, i = 0;
    while (small >= 0 || big >= (temp.size() + 1) / 2) {
        nums[i] = i % 2 ? temp[big--] : temp[small--];
        i++;
    }    
}

---------------------------------------------------------------------
325. Maximum Size Subarray Sum Equals k
Given an array nums and a target value k, find the maximum length of a subarray that sums to k.If there isn't one, return 0 instead.

Note:
The sum of the entire nums array is guaranteed to fit within the 32 - bit signed integer range.

Example 1 :
    Input : nums = [1, -1, 5, -2, 3], k = 3
    Output : 4
    Explanation : The subarray[1, -1, 5, -2] sums to 3 and is the longest.
Example 2 :
    Input : nums = [-2, -1, 2, 1], k = 1
    Output : 2
    Explanation : The subarray[-1, 2] sums to 1 and is the longest.
Follow Up :
Can you do it in O(n) time ?

int maxSubArrayLen(vector<int>& nums, int k) {
    if (nums.empty()) {
        return 0;
    }
    int res = 0;
    map<int, vector<int>> m;
    m[nums[0]].push_back(0);
    vector<int> sum = nums;
    for (int i = 1; i < nums.size(); ++i) {
        sum[i] += sum[i - 1];
        m[sum[i]].push_back(i);
    }
    for (auto it : m) {
        if (it.first == k) {
            res = max(res, it.second.back() + 1);
        }
        else if (m.find(it.first - k) != m.end()) {
            res = max(res, it.second.back() - m[it.first - k][0]);
        }
    }
    return res;
}

---------------------------------------------------------------------
328. Odd Even Linked List
Given a singly linked list, group all odd nodes together followed by the even nodes.Please note here we are talking about the node number and not the value in the nodes.
You should try to do it in place.The program should run in O(1) space complexityand O(nodes) time complexity.

Example 1:
    Input: 1->2->3->4->5->NULL
    Output : 1->3->5->2->4->NULL
Example 2 :
    Input : 2->1->3->5->6->4->7->NULL
    Output : 2->3->6->7->1->5->4->NULL
Note :
    The relative order inside both the even and odd groups should remain as it was in the input.
    The first node is considered odd, the second node even and so on ...

ListNode* oddEvenList(ListNode* head) {
    if (!head || !(head->next)) {
        return head;
    }
    ListNode* a = head, * b = head->next, * enda, * frontb = b;
    while (a && b) {
        a->next = b->next;
        if (a->next) {
            b->next = a->next->next;
            b = b->next;
        }
        enda = a;
        a = a->next;
    }
    if (enda->next) {
        enda->next->next = frontb;
    }
    else {
        enda->next = frontb;
    }
    return head;
}

---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------