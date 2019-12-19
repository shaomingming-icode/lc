156. 上下翻转二叉树
给定一个二叉树，其中所有的右节点要么是具有兄弟节点（拥有相同父节点的左节点）的叶节点，要么为空，将此二叉树上下翻转并将它变成一棵树， 原来的右节点将转换成左叶节点。返回新的根。

例子:

输入: [1,2,3,4,5]

    1
   / \
  2   3
 / \
4   5

输出: 返回二叉树的根 [4,5,2,#,#,3,1]

   4
  / \
 5   2
    / \
   3   1  
说明:

对 [4,5,2,#,#,3,1] 感到困惑? 下面详细介绍请查看 二叉树是如何被序列化的。

二叉树的序列化遵循层次遍历规则，当没有节点存在时，'#' 表示路径终止符。

这里有一个例子:

   1
  / \
 2   3
    /
   4
    \
     5
上面的二叉树则被序列化为 [1,2,3,#,#,4,#,#,5].

TreeNode* upsideDownBinaryTree(TreeNode* root) {        
    if (root == NULL || root->left == NULL) {
        return root;
    }
    TreeNode *res = upsideDownBinaryTree(root->left);
    root->left->left = root->right;
    root->left->right = root;
    root->left = NULL;
    root->right = NULL;
    return res;    
}

------------------------------------------------------------------------------------

159. 至多包含两个不同字符的最长子串
给定一个字符串 s ，找出 至多 包含两个不同字符的最长子串 t 。

示例 1:

输入: "eceba"
输出: 3
解释: t 是 "ece"，长度为3。
示例 2:

输入: "ccaabbb"
输出: 5
解释: t 是 "aabbb"，长度为5。

int lengthOfLongestSubstringTwoDistinct(char * s){
    char twoLetter[2] = { 0, 0 };
    int maxLen = 0, prei = 0, second = 0;
    for (int i = 0; i < strlen(s); i++) {
        prei = i;
        twoLetter[0] = s[i++];
        while (i < strlen(s) && s[i] == twoLetter[0]) {
            i++;
        }
        if (i < strlen(s)) {
            twoLetter[1] = s[i];
            second = i++;
        }
        while (i < strlen(s) && (s[i] == twoLetter[0] || s[i] == twoLetter[1])) {
            i++;
        }
        maxLen = maxLen > i - prei ? maxLen : i - prei;
        if (i < strlen(s)) {
            i = second - 1;
        }
    }
    return maxLen;
}

------------------------------------------------------------------------------------

161. 相隔为 1 的编辑距离
给定两个字符串 s 和 t，判断他们的编辑距离是否为 1。

注意：

满足编辑距离等于 1 有三种可能的情形：

往 s 中插入一个字符得到 t
从 s 中删除一个字符得到 t
在 s 中替换一个字符得到 t
示例 1：

输入: s = "ab", t = "acb"
输出: true
解释: 可以将 'c' 插入字符串 s 来得到 t。
示例 2:

输入: s = "cab", t = "ad"
输出: false
解释: 无法通过 1 步操作使 s 变为 t。
示例 3:

输入: s = "1203", t = "1213"
输出: true
解释: 可以将字符串 s 中的 '0' 替换为 '1' 来得到 t。

bool isInsertDistance(char *first, char *second) {
    for (int i = 0; i < strlen(second); i++) {
        if (first[i] != second[i]) {
            return !strcmp(first + i + 1, second + i);
        }
    }
    return true;
}

bool isReplaceDistance(char *first, char *second) {
    for (int i = 0; i < strlen(second); i++) {
        if (first[i] != second[i]) {
            return !strcmp(first + i + 1, second + i + 1);
        }
    }
    return false;
}

bool isOneEditDistance(char * s, char * t) {
    int lens = strlen(s), lent = strlen(t);
    if ((lens > lent && lens - lent > 1) || (lent > lens && lent - lens > 1)) {
        return false;
    }
    else if (lens > lent && lens - lent == 1) {
        return isInsertDistance(s, t);
    }
    else if (lent > lens && lent - lens == 1) {
        return isInsertDistance(t, s);
    }
    else {
        return isReplaceDistance(s, t);
    }
    return false;
}

------------------------------------------------------------------------------------

163. 缺失的区间
给定一个排序的整数数组 nums ，其中元素的范围在 闭区间 [lower, upper] 当中，返回不包含在数组中的缺失区间。

示例：

输入: nums = [0, 1, 3, 50, 75], lower = 0 和 upper = 99,
输出: ["2", "4->49", "51->74", "76->99"]

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

------------------------------------------------------------------------------------

186. 翻转字符串里的单词 II
给定一个字符串，逐个翻转字符串中的每个单词。

示例：

输入: ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
输出: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]
注意：

单词的定义是不包含空格的一系列字符
输入字符串中不会包含前置或尾随的空格
单词与单词之间永远是以单个空格隔开的
进阶：使用 O(1) 额外空间复杂度的原地解法。

void reverseHelp(char* s, int begin, int end) {
    char temp;
    while (begin < end) {
        temp = s[begin];
        s[begin] = s[end];
        s[end] = temp;
        begin++;
        end--;
    }
}

void reverseWords(char* s, int sSize) {
    if (!s || sSize <= 0) {
        return;
    }
    reverseHelp(s, 0, sSize - 1);
    for (int i = 0; i < sSize; i++) {
        int prebegin = i;
        while (i < sSize && s[i] != ' ') {
            i++;
        }
        reverseHelp(s, prebegin, i - 1);
    }
}

------------------------------------------------------------------------------------

244. 最短单词距离 II
请设计一个类，使该类的构造函数能够接收一个单词列表。然后再实现一个方法，该方法能够分别接收两个单词 word1 和 word2，并返回列表中这两个单词之间的最短距离。您的方法将被以不同的参数调用 多次。

示例:
假设 words = ["practice", "makes", "perfect", "coding", "makes"]

输入: word1 = “coding”, word2 = “practice”
输出: 3
输入: word1 = "makes", word2 = "coding"
输出: 1
注意:
你可以假设 word1 不等于 word2, 并且 word1 和 word2 都在列表里。

class WordDistance {
public:
    WordDistance(vector<string>& words) {
        for (int i = 0; i < words.size(); ++i) {
            m[words[i]].push_back(i);
        }
    }

    int shortest(string word1, string word2) {
        int i = 0, j = 0, res = INT_MAX;
        while (i < m[word1].size() && j < m[word2].size()) {
            res = min(res, abs(m[word1][i] - m[word2][j]));
            m[word1][i] < m[word2][j] ? ++i : ++j;
        }
        return res;
    }
    
private:
    unordered_map<string, vector<int> > m;
};

------------------------------------------------------------------------------------

245. 最短单词距离 III
给定一个单词列表和两个单词 word1 和 word2，返回列表中这两个单词之间的最短距离。

word1 和 word2 是有可能相同的，并且它们将分别表示为列表中两个独立的单词。

示例:
假设 words = ["practice", "makes", "perfect", "coding", "makes"].

输入: word1 = “makes”, word2 = “coding”
输出: 1
输入: word1 = "makes", word2 = "makes"
输出: 3
注意:
你可以假设 word1 和 word2 都在列表里。

int shortestWordDistance(vector<string>& words, string word1, string word2) {
    int p1 = words.size(), p2 = -words.size(), res = INT_MAX;
    for (int i = 0; i < words.size(); ++i) {
        if (words[i] == word1) p1 = word1 == word2 ? p2 : i;
        if (words[i] == word2) p2 = i;
        res = min(res, abs(p1 - p2));
    }
    return res;
}

------------------------------------------------------------------------------------

247. 中心对称数 II
中心对称数是指一个数字在旋转了 180 度之后看起来依旧相同的数字（或者上下颠倒地看）。

找到所有长度为 n 的中心对称数。

示例 :

输入:  n = 2
输出: ["11","69","88","96"]

vector<string> findStrobogrammatic(int n) {
    return find(n, n);
}
vector<string> find(int m, int n) {
    if (m == 0) return {""};
    if (m == 1) return {"0", "1", "8"};
    vector<string> t = find(m - 2, n), res;
    for (auto a : t) {
        if (m != n) res.push_back("0" + a + "0");  // 中间过程要加0
        res.push_back("1" + a + "1");
        res.push_back("6" + a + "9");
        res.push_back("8" + a + "8");
        res.push_back("9" + a + "6");
    }
    return res;
}

------------------------------------------------------------------------------------

249. 移位字符串分组
给定一个字符串，对该字符串可以进行 “移位” 的操作，也就是将字符串中每个字母都变为其在字母表中后续的字母，比如："abc" -> "bcd"。这样，我们可以持续进行 “移位” 操作，从而生成如下移位序列：

"abc" -> "bcd" -> ... -> "xyz"
给定一个包含仅小写字母字符串的列表，将该列表中所有满足 “移位” 操作规律的组合进行分组并返回。

示例：

输入: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]
输出: 
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]

bool isSameWord(char *a, char *b) {
    if (strlen(a) != strlen(b)) {
        return false;
    }
    if (!a || !b) {
        return true;
    }
    while (*(a + 1) != 0 && *(b + 1) != 0) {
        if ((*a + 26 - *(a + 1)) % 26 != (*b + 26 - *(b + 1)) % 26) {
            return false;
        }
        a++;
        b++;
    }
    return true;
}

char *** groupStrings(char ** strings, int stringsSize, int* returnSize, int** returnColumnSizes) {
    int *used = (int*)malloc(sizeof(int) * stringsSize);
    memset(used, 0, sizeof(int) * stringsSize);    
    int *sameWord = (int*)malloc(sizeof(int) * stringsSize * stringsSize);
    memset(sameWord, 0, sizeof(int) * stringsSize * stringsSize);

    int tempcnt = 0;
    for (int i = 0; i < stringsSize; i++) {
        sameWord[i * stringsSize + i] = used[i] == 0 ? 1 : 0;
        for (int j = i + 1; j < stringsSize; j++) {
            if (used[j] == 0 && isSameWord(strings[i], strings[j])) {
                sameWord[i * stringsSize + j] = 1;
                used[j] = 1;
                tempcnt++;
            }
        }
    }

    *returnSize = stringsSize - tempcnt;
    char ***res = (char ***)malloc(sizeof(char**) * *returnSize);
    
    *returnColumnSizes = (int*)malloc(sizeof(int) * *returnSize);
    int colIndex = 0;
    for (int i = 0; i < stringsSize; i++) {
        int temp = 0;
        for (int j = 0; j < stringsSize; j++) {
            temp += sameWord[i * stringsSize + j];
        }
        if (temp != 0) {
            (*returnColumnSizes)[colIndex] = temp;
            res[colIndex] = (char **)malloc(sizeof(char*) * temp);
            int coltempIndex = 0;
            for (int j = 0; j < stringsSize; j++) {
                if (sameWord[i * stringsSize + j] == 1) {
                    res[colIndex][coltempIndex] = strings[j];
                    coltempIndex++;
                }
            }
            colIndex++;
        }
    }
    free(used);
    free(sameWord);
    return res;
}

------------------------------------------------------------------------------------

250. 统计同值子树
给定一个二叉树，统计该二叉树数值相同的子树个数。

同值子树是指该子树的所有节点都拥有相同的数值。

示例：

输入: root = [5,1,5,5,5,null,5]

              5
             / \
            1   5
           / \   \
          5   5   5

输出: 4

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */

bool helpSubtrees(struct TreeNode* root, int val, int* res) {
    if (!root) {
        return true;
    }
    bool left = helpSubtrees(root->left, root->val, res);
    bool right = helpSubtrees(root->right, root->val, res);
    if (!left || !right) {
        return false;
    }
    (*res)++;
    return root->val == val;
}

int countUnivalSubtrees(struct TreeNode* root){
    int res = 0;
    helpSubtrees(root, -1, &res);
    return res;
}

------------------------------------------------------------------------------------

251. 展开二维向量
请设计并实现一个能够展开二维向量的迭代器。该迭代器需要支持 next 和 hasNext 两种操作。、

示例：

Vector2D iterator = new Vector2D([[1,2],[3],[4]]);

iterator.next(); // 返回 1
iterator.next(); // 返回 2
iterator.next(); // 返回 3
iterator.hasNext(); // 返回 true
iterator.hasNext(); // 返回 true
iterator.next(); // 返回 4
iterator.hasNext(); // 返回 false
 

注意：

请记得 重置 在 Vector2D 中声明的类变量（静态变量），因为类变量会 在多个测试用例中保持不变，影响判题准确。请 查阅 这里。
你可以假定 next() 的调用总是合法的，即当 next() 被调用时，二维向量总是存在至少一个后续元素。
 

进阶：

尝试在代码中仅使用 C++ 提供的迭代器 或 Java 提供的迭代器。

typedef struct {
    int *oneLine;
    int index;
    int size;
} Vector2D;


Vector2D* vector2DCreate(int** v, int vSize, int* vColSize) {
    int len = 0;
    for(int i = 0; i < vSize; i++) {
        len += vColSize[i];
    }
    Vector2D *res = (Vector2D*)malloc(sizeof(Vector2D));
    res->oneLine = (int*)malloc(sizeof(int) * len);
    res->index = 0;
    res->size = len;
    for(int i = 0; i < vSize; i++) {
        for(int j = 0; j < vColSize[i]; j++) {
            res->oneLine[res->index++] = v[i][j];
        }
    }
    res->index = 0;
    return res;
}

int vector2DNext(Vector2D* obj) {
    return obj->oneLine[obj->index++];
}

bool vector2DHasNext(Vector2D* obj) {
    return obj->index < obj->size;
}

void vector2DFree(Vector2D* obj) {
    free(obj->oneLine);
    free(obj);
}

/**
 * Your Vector2D struct will be instantiated and called as such:
 * Vector2D* obj = vector2DCreate(v, vSize, vColSize);
 * int param_1 = vector2DNext(obj);
 
 * bool param_2 = vector2DHasNext(obj);
 
 * vector2DFree(obj);
*/

------------------------------------------------------------------------------------

253. 会议室 II
给定一个会议时间安排的数组，每个会议时间都会包括开始和结束的时间 [[s1,e1],[s2,e2],...] (si < ei)，为避免会议冲突，同时要考虑充分利用会议室资源，请你计算至少需要多少间会议室，才能满足这些会议安排。

示例 1:

输入: [[0, 30],[5, 10],[15, 20]]
输出: 2
示例 2:

输入: [[7,10],[2,4]]
输出: 1

int compare(const void *a, const void *b) {
    return *((int*)a) - *((int*)b);
}

int minMeetingRooms(int** intervals, int intervalsSize, int* intervalsColSize) {
	int *begin = (int*)malloc(sizeof(int) * intervalsSize);
    int *end = (int*)malloc(sizeof(int) * intervalsSize);

	for(int i = 0; i < intervalsSize; i++) {
        begin[i] = intervals[i][0];
        end[i] = intervals[i][1];
    }

    qsort(begin, intervalsSize, sizeof(int), compare);
    qsort(end, intervalsSize, sizeof(int), compare);
    
    int res = 0, roomNum = 0, beginIndex = 0, endIndex = 0;
    while(beginIndex < intervalsSize && endIndex < intervalsSize) {
        if(begin[beginIndex] < end[endIndex]) {
            roomNum++;
            beginIndex++;
        }
        else {
            roomNum--;
            endIndex++;
        }
        res = res > roomNum ? res : roomNum;
    }
    return res;
}

------------------------------------------------------------------------------------

254. 因子的组合
整数可以被看作是其因子的乘积。

例如：

8 = 2 x 2 x 2;
  = 2 x 4.
请实现一个函数，该函数接收一个整数 n 并返回该整数所有的因子组合。

注意：

你可以假定 n 为永远为正数。
因子必须大于 1 并且小于 n。
示例 1：

输入: 1
输出: []
示例 2：

输入: 37
输出: []
示例 3：

输入: 12
输出:
[
  [2, 6],
  [2, 2, 3],
  [3, 4]
]
示例 4:

输入: 32
输出:
[
  [2, 16],
  [2, 2, 8],
  [2, 2, 2, 4],
  [2, 2, 2, 2, 2],
  [2, 4, 4],
  [4, 8]
]

vector<vector<int>> getFactors(int n) {
        vector<vector<int>> res;
        helper(n, 2, {}, res);
        return res;
    }
    void helper(int n, int start, vector<int> out, vector<vector<int>> &res) {
        for (int i = start; i <= sqrt(n); i++) {
            if (n % i == 0) {
                vector<int> new_out = out;
                new_out.push_back(i);
                helper(n / i, i, new_out, res);
                new_out.push_back(n / i);
                res.push_back(new_out);
            }
        }
    }

------------------------------------------------------------------------------------

255. 验证前序遍历序列二叉搜索树
给定一个整数数组，你需要验证它是否是一个二叉搜索树正确的先序遍历序列。

你可以假定该序列中的数都是不相同的。

参考以下这颗二叉搜索树：

     5
    / \
   2   6
  / \
 1   3
示例 1：

输入: [5,2,6,1,3]
输出: false
示例 2：

输入: [5,2,1,3,6]
输出: true
进阶挑战：

您能否使用恒定的空间复杂度来完成此题？

bool verifyPreorder(vector<int>& preorder) {
    return helper(preorder, 0, preorder.size() - 1, INT_MIN, INT_MAX);
}
bool helper(vector<int>& preorder, int start, int end, int lower, int upper) {
    if (start > end) return true;
    int val = preorder[start], i = 0;
    if (val <= lower || val >= upper) return false;
    for (i = start + 1; i <= end; ++i) {
        if (preorder[i] >= val) break;
    }
    return helper(preorder, start + 1, i - 1, lower, val) && helper(preorder, i, end, val, upper);
}

------------------------------------------------------------------------------------

259. 较小的三数之和
给定一个长度为 n 的整数数组和一个目标值 target，寻找能够使条件 nums[i] + nums[j] + nums[k] < target 成立的三元组  i, j, k 个数（0 <= i < j < k < n）。

示例：

输入: nums = [-2,0,1,3], target = 2
输出: 2 
解释: 因为一共有两个三元组满足累加和小于 2:
     [-2,0,1]
     [-2,0,3]
进阶：是否能在 O(n2) 的时间复杂度内解决？

int threeSumSmaller(vector<int>& nums, int target) {
    if (nums.size() < 3) return 0;
    int res = 0, n = nums.size();
    sort(nums.begin(), nums.end());
    for (int i = 0; i < n - 2; ++i) {
        int left = i + 1, right = n - 1;
        while (left < right) {
            if (nums[i] + nums[left] + nums[right] < target) {
                res += right - left;
                ++left;
            } else {
                --right;
            }
        }
    }
    return res;
}

------------------------------------------------------------------------------------

261. 以图判树
给定从 0 到 n-1 标号的 n 个结点，和一个无向边列表（每条边以结点对来表示），请编写一个函数用来判断这些边是否能够形成一个合法有效的树结构。

示例 1：
输入: n = 5, 边列表 edges = [[0,1], [0,2], [0,3], [1,4]]
输出: true

示例 2:
输入: n = 5, 边列表 edges = [[0,1], [1,2], [2,3], [1,3], [1,4]]
输出: false
注意：你可以假定边列表 edges 中不会出现重复的边。由于所有的边是无向边，边 [0,1] 和边 [1,0] 是相同的，因此不会同时出现在边列表 edges 中。

bool validTree(int n, vector<vector<int>>& edges) {
    vector<vector<int>> graph(n, vector<int>());
    vector<bool> flag(n, false);
    for (auto a : edges) {
        graph[a[0]].push_back(a[1]);
        graph[a[1]].push_back(a[0]);
    }
    if (!dfs(graph, flag, 0, -1)) {  // 环
        return false;
    }
    for (auto a : flag) {  // 连通
        if (!a) {
            return false;
        }
    }
    return true;
}

bool dfs(vector<vector<int>> &graph, vector<bool> &flag, int cur, int pre) {
    if (flag[cur]) {
        return false;
    }
    flag[cur] = true;
    for (auto a : graph[cur]) {
        if (a != pre) {
            if (!dfs(graph, flag, a, cur)) {
                return false;
            }
        }
    }
    return true;
}

------------------------------------------------------------------------------------

267. 回文排列 II
给定一个字符串 s ，返回其通过重新排列组合后所有可能的回文字符串，并去除重复的组合。

如不能形成任何回文排列时，则返回一个空列表。

示例 1：

输入: "aabb"
输出: ["abba", "baab"]
示例 2：

输入: "abc"
输出: []

vector<string> generatePalindromes(string s) {
    unordered_set<string> res;
    unordered_map<char, int> m;
    string half = "", mid = "";
    for (auto a : s) {
        m[a]++;
    }
    for (auto it : m) {
        if (it.second % 2 == 1) {
            mid += it.first;
        }
        half += string(it.second / 2, it.first);
        if (mid.size() > 1) {
            return {};
        }
    }
    permute(half, 0, mid, res);
    return vector<string>(res.begin(), res.end());
}

void permute(string &half, int start, string mid, unordered_set<string> &res) {
    if (start == half.size()) {
        res.insert(half + mid + string(half.rbegin(), half.rend()));
    }
    for (int i = start; i < half.size(); ++i) {
        if (i != start && half[i] == half[start]) {
            continue;
        }
        swap(half[i], half[start]);
        permute(half, start + 1, mid, res);
        swap(half[i], half[start]);
    }
}

------------------------------------------------------------------------------------

271. 字符串的编码与解码
请你设计一个算法，可以将一个 字符串列表 编码成为一个 字符串。这个编码后的字符串是可以通过网络进行高效传送的，并且可以在接收端被解码回原来的字符串列表。

1 号机（发送方）有如下函数：

string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}
2 号机（接收方）有如下函数：

vector<string> decode(string s) {
  //... your code
  return strs;
}
1 号机（发送方）执行：

string encoded_string = encode(strs);
2 号机（接收方）执行：

vector<string> strs2 = decode(encoded_string);
此时，2 号机（接收方）的 strs2 需要和 1 号机（发送方）的 strs 相同。

请你来实现这个 encode 和 decode 方法。

注意：

因为字符串可能会包含 256 个合法 ascii 字符中的任何字符，所以您的算法必须要能够处理任何可能会出现的字符。
请勿使用 “类成员”、“全局变量” 或 “静态变量” 来存储这些状态，您的编码和解码算法应该是非状态依赖的。
请不要依赖任何方法库，例如 eval 又或者是 serialize 之类的方法。本题的宗旨是需要您自己实现 “编码” 和 “解码” 算法。

/** Encodes a list of strings to a single string */
char* encode(char** strs, int strsSize) {
    int len = 0;
    for (int i = 0; i < strsSize; i++) {
        len += strlen(strs[i]) + sizeof(int);
    }
    char* res = (char*)malloc(sizeof(char) * (len + 1));
    int resIndex = 0;
    for (int i = 0; i < strsSize; i++) {
        int len = strlen(strs[i]);
        res[resIndex++] = *((char*)(&len) + 3) + '0';
        res[resIndex++] = *((char*)(&len) + 2) + '0';
        res[resIndex++] = *((char*)(&len) + 1) + '0';
        res[resIndex++] = *((char*)(&len)) + '0';
        strcpy(res + resIndex, strs[i]);
        resIndex += strlen(strs[i]);
    }
    res[len] = 0;
    return res;
}

/**
 * Decodes a single string to a list of strings.
 *
 * Return an array of size *returnSize.
 * Note: The returned array must be malloced, assume caller calls free().
 */
char** decode(char* s, int* returnSize) {
    *returnSize = 0;
    int len = 0;
    int temps = 0;
    while (temps < strlen(s)) {
        len = 0;
        len += (*(s + temps) - '0') << 24;
        len += (*(s + temps + 1) - '0') << 16;
        len += (*(s + temps + 2) - '0') << 8;
        len += *(s + temps + 3) - '0';
        temps += 4 + len;
        (*returnSize)++;
    }
    char** res = (char**)malloc(sizeof(char*) * *returnSize);
    len = 0;
    temps = 0;
    for (int i = 0; i < *returnSize; i++) {
        len = 0;
        len += (*(s + temps) - '0') << 24;
        len += (*(s + temps + 1) - '0') << 16;
        len += (*(s + temps + 2) - '0') << 8;
        len += *(s + temps + 3) - '0';
        res[i] = (char*)malloc(sizeof(char) * (len + 1));
        strncpy(res[i], (s + temps + 4), len);
        res[i][len] = 0;
        temps += 4 + len;
    }
    return res;
}

------------------------------------------------------------------------------------

277. 搜寻名人
假设你是一个专业的狗仔，参加了一个 n 人派对，其中每个人被从 0 到 n - 1 标号。在这个派对人群当中可能存在一位 “名人”。所谓 “名人” 的定义是：其他所有 n - 1 个人都认识他/她，而他/她并不认识其他任何人。

现在你想要确认这个 “名人” 是谁，或者确定这里没有 “名人”。而你唯一能做的就是问诸如 “A 你好呀，请问你认不认识 B呀？” 的问题，以确定 A 是否认识 B。你需要在（渐近意义上）尽可能少的问题内来确定这位 “名人” 是谁（或者确定这里没有 “名人”）。

在本题中，你可以使用辅助函数 bool knows(a, b) 获取到 A 是否认识 B。请你来实现一个函数 int findCelebrity(n)。

派对最多只会有一个 “名人” 参加。若 “名人” 存在，请返回他/她的编号；若 “名人” 不存在，请返回 -1。

示例 1:

输入: graph = [
  [1,1,0],
  [0,1,0],
  [1,1,1]
]
输出: 1
解析: 有编号分别为 0、1 和 2 的三个人。graph[i][j] = 1 代表编号为 i 的人认识编号为 j 的人，而 graph[i][j] = 0 则代表编号为 i 的人不认识编号为 j 的人。“名人” 是编号 1 的人，因为 0 和 2 均认识他/她，但 1 不认识任何人。
示例 2:

输入: graph = [
  [1,0,1],
  [1,1,0],
  [0,1,1]
]
输出: -1
解析: 没有 “名人”
 
注意:

该有向图是以邻接矩阵的形式给出的，是一个 n × n 的矩阵， a[i][j] = 1 代表 i 与 j 认识，a[i][j] = 0 则代表 i 与 j 不认识。
请记住，您是无法直接访问邻接矩阵的。

int findCelebrity(int n) {
    int* celebrity = (int*)malloc(sizeof(int) * n);
    memset(celebrity, 0, sizeof(int) * n);
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && knows(i, j) == 1) {
                break;
            }
        }
        if (j == n) {
            celebrity[i] = 1;
        }
    }
    for (i = 0; i < n; i++) {
        if (celebrity[i] == 1) {
            for (j = 0; j < n; j++) {
                if (i != j && knows(j, i) == 0) {
                    break;
                }
            }
            if (j == n) {
                free(celebrity);
                return i;
            }
        }
    }
    free(celebrity);
    return -1;
}

------------------------------------------------------------------------------------

280. 摆动排序
给你一个无序的数组 nums, 将该数字 原地 重排后使得 nums[0] <= nums[1] >= nums[2] <= nums[3]...。

示例:

输入: nums = [3,5,2,1,6,4]
输出: 一个可能的解答是 [3,5,1,6,2,4]

void swapInt(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void wiggleSort(int* nums, int numsSize) {
    if (!nums || numsSize <= 0) {
        return;
    }
    for (int i = 1; i < numsSize; i++) {
        if ((i % 2 == 0 && nums[i] > nums[i - 1]) ||
            (i % 2 == 1 && nums[i] < nums[i - 1])) {
            swapInt(&nums[i], &nums[i - 1]);
        }
    }
}

------------------------------------------------------------------------------------

281. 锯齿迭代器
给出两个一维的向量，请你实现一个迭代器，交替返回它们中间的元素。

示例:

输入:
v1 = [1,2]
v2 = [3,4,5,6] 

输出: [1,3,2,4,5,6]

解析: 通过连续调用 next 函数直到 hasNext 函数返回 false，
     next 函数返回值的次序应依次为: [1,3,2,4,5,6]。
拓展：假如给你 k 个一维向量呢？你的代码在这种情况下的扩展性又会如何呢?

拓展声明：
 “锯齿” 顺序对于 k > 2 的情况定义可能会有些歧义。所以，假如你觉得 “锯齿” 这个表述不妥，也可以认为这是一种 “循环”。例如：

输入:
[1,2,3]
[4,5,6,7]
[8,9]

输出: [1,4,8,2,5,9,3,6,7].

struct ZigzagIterator {
    int *v1;
    int v1Size;
    int v1Index;
    int *v2;
    int v2Size;
    int v2Index;
};

struct ZigzagIterator *zigzagIteratorCreate(int* v1, int v1Size, int* v2, int v2Size) {
    struct ZigzagIterator *res = (struct ZigzagIterator*)malloc(sizeof(struct ZigzagIterator));
    res->v1 = v1;
    res->v1Size = v1Size;
    res->v1Index = 0;
    res->v2 = v2;
    res->v2Size = v2Size;
    res->v2Index = 0;
    return res;
}

bool zigzagIteratorHasNext(struct ZigzagIterator *iter) {
    return iter->v1Index < iter->v1Size || iter->v2Index < iter->v2Size;
}

int zigzagIteratorNext(struct ZigzagIterator *iter) {
    if (iter->v1Index == iter->v1Size) {
        return iter->v2[iter->v2Index++];
    }
    else if (iter->v2Index == iter->v2Size) {
        return iter->v1[iter->v1Index++];
    }
    else if (iter->v1Index < iter->v1Size && iter->v1Index <= iter->v2Index) {
        return iter->v1[iter->v1Index++];
    }
    else if (iter->v2Index < iter->v2Size && iter->v1Index > iter->v2Index) {
        return iter->v2[iter->v2Index++];
    }
    return 0;
}

/** Deallocates memory previously allocated for the iterator */
void zigzagIteratorFree(struct ZigzagIterator *iter) {
    free(iter);
}

/**
 * Your ZigzagIterator will be called like this:
 * struct ZigzagIterator *i = zigzagIteratorCreate(v1, v1Size, v2, v2Size);
 * while (zigzagIteratorHasNext(i)) printf("%d\n", zigzagIteratorNext(i));
 * zigzagIteratorFree(i);
 */
 
------------------------------------------------------------------------------------

285. 二叉搜索树中的顺序后继
给你一个二叉搜索树和其中的某一个结点，请你找出该结点在树中顺序后继的节点。

结点 p 的后继是值比 p.val 大的结点中键值最小的结点。

 

示例 1:



输入: root = [2,1,3], p = 1
输出: 2
解析: 这里 1 的顺序后继是 2。请注意 p 和返回值都应是 TreeNode 类型。
示例 2:



输入: root = [5,3,6,2,4,null,null,1], p = 6
输出: null
解析: 因为给出的结点没有顺序后继，所以答案就返回 null 了。
 

注意:

假如给出的结点在该树中没有顺序后继的话，请返回 null
我们保证树中每个结点的值是唯一的

struct TreeNode* inorderSuccessor(struct TreeNode* root, struct TreeNode* p) {
    if (!root) {
        return NULL;
    }
    if (root->val <= p->val) {
        return inorderSuccessor(root->right, p);
    } else {
        struct TreeNode *left = inorderSuccessor(root->left, p);
        return left ? left : root;
    }   
}

------------------------------------------------------------------------------------

286. 墙与门
你被给定一个 m × n 的二维网格，网格中有以下三种可能的初始化值：

-1 表示墙或是障碍物
0 表示一扇门
INF 无限表示一个空的房间。然后，我们用 231 - 1 = 2147483647 代表 INF。你可以认为通往门的距离总是小于 2147483647 的。
你要给每个空房间位上填上该房间到 最近 门的距离，如果无法到达门，则填 INF 即可。

示例：

给定二维网格：

INF  -1  0  INF
INF INF INF  -1
INF  -1 INF  -1
  0  -1 INF INF
运行完你的函数后，该网格应该变成：

  3  -1   0   1
  2   2   1  -1
  1  -1   2  -1
  0  -1   3   4

void wallsAndGates(int** rooms, int roomsSize, int* roomsColSize){
    for (int i = 0; i < roomsSize; i++) {
        for (int j = 0; j < roomsColSize[0]; j++) {
            if (rooms[i][j] == 0) {
                dfs(rooms, roomsSize, roomsColSize[0], i, j, 0);
            }
        }
    }
}

void dfs(int **rooms, int size, int colSize, int i, int j, int val) {
    if (i < 0 || i >= size || j < 0 || j >= colSize || rooms[i][j] < val) {
        return;
    }
    rooms[i][j] = val;
    dfs(rooms, size, colSize, i + 1, j, val + 1);
    dfs(rooms, size, colSize, i - 1, j, val + 1);
    dfs(rooms, size, colSize, i, j + 1, val + 1);
    dfs(rooms, size, colSize, i, j - 1, val + 1);
}

------------------------------------------------------------------------------------

288. 单词的唯一缩写
一个单词的缩写需要遵循 <起始字母><中间字母数><结尾字母> 这样的格式。

以下是一些单词缩写的范例：

a) it                      --> it    (没有缩写)

     1
     ↓
b) d|o|g                   --> d1g

              1    1  1
     1---5----0----5--8
     ↓   ↓    ↓    ↓  ↓    
c) i|nternationalizatio|n  --> i18n

              1
     1---5----0
     ↓   ↓    ↓
d) l|ocalizatio|n          --> l10n
假设你有一个字典和一个单词，请你判断该单词的缩写在这本字典中是否唯一。若单词的缩写在字典中没有任何 其他 单词与其缩写相同，则被称为单词的唯一缩写。

示例：

给定 dictionary = [ "deer", "door", "cake", "card" ]

isUnique("dear") -> false
isUnique("cart") -> true
isUnique("cane") -> false
isUnique("make") -> true

typedef struct {
    char **words;
    char **dictionary;
    int size;
} ValidWordAbbr;
int getStrLen(int n) {
    int len = 0;
    while (n) {
        len++;
        n /= 10;
    }
    return len;
}
void addNewSize(char *str, int n, int size) {
    while (n) {
        *(str + size - 1) = n % 10;
        size--;
        n /= 10;
    }
}
ValidWordAbbr* validWordAbbrCreate(char ** dictionary, int dictionarySize) {
    ValidWordAbbr *res = (ValidWordAbbr*)malloc(sizeof(ValidWordAbbr));
    res->words = (char**)malloc(sizeof(char*) * dictionarySize);
    res->size = dictionarySize;
    res->dictionary = dictionary;
    for (int i = 0; i < dictionarySize; i++) {
        if (strlen(dictionary[i]) > 2) {
            int dirLen = strlen(dictionary[i]);
            int newSize = getStrLen(dirLen - 2) + 2;
            res->words[i] = (char*)malloc(sizeof(char) * (newSize + 1));
            res->words[i][0] = dictionary[i][0];
            addNewSize(res->words[i] + 1, dirLen - 2, newSize);
            res->words[i][newSize - 1] = dictionary[i][dirLen - 1];
            res->words[i][newSize] = 0;
        }
        else {
            res->words[i] = NULL;
        }
    }
    return res;
}
bool validWordAbbrIsUnique(ValidWordAbbr* obj, char * word) {
    if (!word || strlen(word) <= 2) {
        return true;
    }
    int wordLen = strlen(word);
    int newSize = getStrLen(wordLen - 2) + 2;
    char *temp = (char*)malloc(sizeof(char) * (newSize + 1));
    temp[0] = word[0];
    addNewSize(temp + 1, wordLen - 2, newSize);
    temp[newSize - 1] = word[wordLen - 1];
    temp[newSize] = 0;
    for (int i = 0; i < obj->size; i++) {
        if (obj->words[i] != NULL && strcmp(temp, obj->words[i]) == 0 && strcmp(word, obj->dictionary[i]) != 0) {
            return false;
        }
    }
    free(temp);
    return true;
}

void validWordAbbrFree(ValidWordAbbr* obj) {
    for (int i = 0; i < obj->size; i++) {
        free(obj->words[i]);
    }
    free(obj);
}

/**
 * Your ValidWordAbbr struct will be instantiated and called as such:
 * ValidWordAbbr* obj = validWordAbbrCreate(dictionary, dictionarySize);
 * bool param_1 = validWordAbbrIsUnique(obj, word);
 
 * validWordAbbrFree(obj);
*/

------------------------------------------------------------------------------------

294. 翻转游戏 II
你和朋友玩一个叫做「翻转游戏」的游戏，游戏规则：给定一个只有 + 和 - 的字符串。你和朋友轮流将 连续 的两个 "++" 反转成 "--"。 当一方无法进行有效的翻转时便意味着游戏结束，则另一方获胜。

请你写出一个函数来判定起始玩家是否存在必胜的方案。

示例：

输入: s = "++++"
输出: true 
解析: 起始玩家可将中间的 "++" 翻转变为 "+--+" 从而得胜。
延伸：
请推导你算法的时间复杂度。

bool canWin(char * s){
    for (int i = 1; i < strlen(s); i++) {
        if (s[i] == '+' && s[i - 1] == '+') {
            s[i] = s[i - 1] = '-';
            if (!canWin(s)) {
                s[i] = s[i - 1] = '+';
                return true;
            }
            s[i] = s[i - 1] = '+';
        }
    }
    return false;
}

------------------------------------------------------------------------------------

298. 二叉树最长连续序列
给你一棵指定的二叉树，请你计算它最长连续序列路径的长度。

该路径，可以是从某个初始结点到树中任意结点，通过「父 - 子」关系连接而产生的任意路径。

这个最长连续的路径，必须从父结点到子结点，反过来是不可以的。

示例 1：

输入:

   1
    \
     3
    / \
   2   4
        \
         5

输出: 3

解析: 当中，最长连续序列是 3-4-5，所以返回结果为 3
示例 2：

输入:

   2
    \
     3
    / 
   2    
  / 
 1

输出: 2 

解析: 当中，最长连续序列是 2-3。注意，不是 3-2-1，所以返回 2。

void helpCons(struct TreeNode* root, int parentVal, int thisres, int *res) {
    if (root == NULL) {
        return;
    }
    if (root->val != parentVal + 1) {
        helpCons(root->left, root->val, 1, res);
        helpCons(root->right, root->val, 1, res);
    }
    else {
        thisres++;
        *res = thisres > *res ? thisres : *res;
        helpCons(root->left, root->val, thisres, res);
        helpCons(root->right, root->val, thisres, res);
    }
}

int longestConsecutive(struct TreeNode* root) {
    if (root == NULL) {
        return 0;
    }
    int res = 1;
    helpCons(root, root->val - 1, 0, &res);
    return res;
}

------------------------------------------------------------------------------------

311. 稀疏矩阵的乘法
给定两个 稀疏矩阵 A 和 B，请你返回 AB。你可以默认 A 的列数等于 B 的行数。

请仔细阅读下面的示例。
示例：

输入:
A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]
B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]

输出:
     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                  | 0 0 1 |

int** multiply(int** A, int ASize, int* AColSize, int** B, int BSize, int* BColSize, int* returnSize, int** returnColumnSizes){
    *returnSize = ASize;
    *returnColumnSizes = (int*)malloc(sizeof(int) * *returnSize);
    for (int i = 0; i < *returnSize; i++) {
        (*returnColumnSizes)[i] = BColSize[0];
    }
    int **res = (int**)malloc(sizeof(int*) * *returnSize);
    for (int i = 0; i < *returnSize; i++) {
        res[i] = malloc(sizeof(int) * (*returnColumnSizes)[i]);
    }
    for (int i = 0; i < *returnSize; i++) {
        for (int j = 0; j < (*returnColumnSizes)[0]; j++) {
            int sum = 0;
            for (int k = 0; k < AColSize[0]; k++) {
                sum += A[i][k] * B[k][j];
            }
            res[i][j] = sum;
        }
    }
    return res;
}

------------------------------------------------------------------------------------

314	二叉树的垂直遍历
给定一个二叉树，返回其结点 垂直方向（从上到下，逐列）遍历的值。

如果两个结点在同一行和列，那么顺序则为 从左到右。

示例 1：

输入: [3,9,20,null,null,15,7]

   3
  /\
 /  \
9   20
    /\
   /  \
  15   7 

输出:

[
  [9],
  [3,15],
  [20],
  [7]
]
示例 2:

输入: [3,9,8,4,0,1,7]

     3
    /\
   /  \
  9    8
  /\   /\
 /  \ /  \
4   0 1   7 

输出:

[
  [4],
  [9],
  [3,0,1],
  [8],
  [7]
]
示例 3:

输入: [3,9,8,4,0,1,7,null,null,null,2,5]（注意：0 的右侧子节点为 2，1 的左侧子节点为 5）

     3
    /\
   /  \
   9   8
  /\  /\
 /  \/  \
 4  01   7
    /\
   /  \
   5   2

输出:

[
  [4],
  [9,5],
  [3,0,1],
  [8,2],
  [7]
]

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> verticalOrder(TreeNode* root) {
        vector<vector<int>> res;
        map<int, vector<int>> hOrder;
        queue<pair<int, TreeNode*>> nodes;
        nodes.push(make_pair(0, root));

        while (nodes.size() != 0) {
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes.front().second == NULL) {
                    nodes.pop();
                    continue;
                }
                hOrder[nodes.front().first].push_back(nodes.front().second->val);
                nodes.push(make_pair(nodes.front().first - 1, nodes.front().second->left));
                nodes.push(make_pair(nodes.front().first + 1, nodes.front().second->right));
                nodes.pop();
            }
        }
        for (auto it = hOrder.begin(); it != hOrder.end(); it++) {
            res.push_back(it->second);
        }
        return res;
    }
};

------------------------------------------------------------------------------------

320	列举单词的全部缩写
请你写出一个能够举单词全部缩写的函数。

注意：输出的顺序并不重要。

示例：

输入: "word"
输出:
["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

class Solution {
public:
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
};

------------------------------------------------------------------------------------

323	无向图中连通分量的数目
给定编号从 0 到 n-1 的 n 个节点和一个无向边列表（每条边都是一对节点），请编写一个函数来计算无向图中连通分量的数目。

示例 1:

输入: n = 5 和 edges = [[0, 1], [1, 2], [3, 4]]

     0          3
     |          |
     1 --- 2    4 

输出: 2
示例 2:

输入: n = 5 和 edges = [[0, 1], [1, 2], [2, 3], [3, 4]]

     0           4
     |           |
     1 --- 2 --- 3

输出:  1
注意:
你可以假设在 edges 中不会出现重复的边。而且由于所以的边都是无向边，[0, 1] 与 [1, 0]  相同，所以它们不会同时在 edges 中出现。

class Solution {
public:
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
};

------------------------------------------------------------------------------------

325	和等于 k 的最长子数组长度
给定一个数组 nums 和一个目标值 k，找到和等于 k 的最长子数组长度。如果不存在任意一个符合要求的子数组，则返回 0。

注意:
 nums 数组的总和是一定在 32 位有符号整数范围之内的。

示例 1:

输入: nums = [1, -1, 5, -2, 3], k = 3
输出: 4 
解释: 子数组 [1, -1, 5, -2] 和等于 3，且长度最长。
示例 2:

输入: nums = [-2, -1, 2, 1], k = 1
输出: 2 
解释: 子数组 [-1, 2] 和等于 1，且长度最长。
进阶:
你能使时间复杂度在 O(n) 内完成此题吗?

class Solution {
public:
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
};

------------------------------------------------------------------------------------

333 最大 BST 子树
给定一个二叉树，找到其中最大的二叉搜索树（BST）子树，其中最大指的是子树节点数最多的。

注意:
子树必须包含其所有后代。

示例:

输入: [10,5,15,1,8,null,7]

   10 
   / \ 
  5  15 
 / \   \ 
1   8   7

输出: 3
解释: 高亮部分为最大的 BST 子树。
     返回值 3 在这个样例中为子树大小。
进阶:
你能想出用 O(n) 的时间复杂度解决这个问题吗？

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */

int largestBSTSubtreeHelper(struct TreeNode* root, int *minValue, int *maxValue)
{
    if (!root) {
        *minValue = INT_MAX;
        *maxValue = INT_MIN;
        return 0;
    }
    int minLeftValue, maxLeftValue, minRightValue, maxRightValue;
    int left = largestBSTSubtreeHelper(root->left, &minLeftValue, &maxLeftValue);
    int right = largestBSTSubtreeHelper(root->right, &minRightValue, &maxRightValue);
    if (root->val > maxLeftValue && root->val < minRightValue) {
        *minValue = root->val < minLeftValue ? root->val : minLeftValue;
        *maxValue = root->val > maxRightValue ? root->val : maxRightValue;
        return left + right + 1;
    }
    else {
        *minValue = INT_MIN;
        *maxValue = INT_MAX;
        return left > right ? left : right;
    }
}

int largestBSTSubtree(struct TreeNode* root) {
    int minValue = 0, maxValue = 0;
    return largestBSTSubtreeHelper(root, &minValue, &maxValue);
}

------------------------------------------------------------------------------------

348	判定井字棋胜负
请在 n × n 的棋盘上，实现一个判定井字棋（Tic-Tac-Toe）胜负的神器，判断每一次玩家落子后，是否有胜出的玩家。

在这个井字棋游戏中，会有 2 名玩家，他们将轮流在棋盘上放置自己的棋子。

在实现这个判定器的过程中，你可以假设以下这些规则一定成立：

      1. 每一步棋都是在棋盘内的，并且只能被放置在一个空的格子里；

      2. 一旦游戏中有一名玩家胜出的话，游戏将不能再继续；

      3. 一个玩家如果在同一行、同一列或者同一斜对角线上都放置了自己的棋子，那么他便获得胜利。

示例:

给定棋盘边长 n = 3, 玩家 1 的棋子符号是 "X"，玩家 2 的棋子符号是 "O"。

TicTacToe toe = new TicTacToe(3);

toe.move(0, 0, 1); -> 函数返回 0 (此时，暂时没有玩家赢得这场对决)
|X| | |
| | | |    // 玩家 1 在 (0, 0) 落子。
| | | |

toe.move(0, 2, 2); -> 函数返回 0 (暂时没有玩家赢得本场比赛)
|X| |O|
| | | |    // 玩家 2 在 (0, 2) 落子。
| | | |

toe.move(2, 2, 1); -> 函数返回 0 (暂时没有玩家赢得比赛)
|X| |O|
| | | |    // 玩家 1 在 (2, 2) 落子。
| | |X|

toe.move(1, 1, 2); -> 函数返回 0 (暂没有玩家赢得比赛)
|X| |O|
| |O| |    // 玩家 2 在 (1, 1) 落子。
| | |X|

toe.move(2, 0, 1); -> 函数返回 0 (暂无玩家赢得比赛)
|X| |O|
| |O| |    // 玩家 1 在 (2, 0) 落子。
|X| |X|

toe.move(1, 0, 2); -> 函数返回 0 (没有玩家赢得比赛)
|X| |O|
|O|O| |    // 玩家 2 在 (1, 0) 落子.
|X| |X|

toe.move(2, 1, 1); -> 函数返回 1 (此时，玩家 1 赢得了该场比赛)
|X| |O|
|O|O| |    // 玩家 1 在 (2, 1) 落子。
|X|X|X|
 

进阶:
您有没有可能将每一步的 move() 操作优化到比 O(n2) 更快吗?

typedef struct {
    int* row[2];
    int* col[2];
    int diagonalLeft[2];
    int diagonalRight[2];
    int n;
} TicTacToe;

/** Initialize your data structure here. */

TicTacToe* ticTacToeCreate(int n) {
    TicTacToe *res = (TicTacToe*)malloc(sizeof(TicTacToe));
    res->row[0] = (int*)malloc(sizeof(int) * n);
    memset(res->row[0], 0, sizeof(int) * n);
    res->row[1] = (int*)malloc(sizeof(int) * n);
    memset(res->row[1], 0, sizeof(int) * n);
    res->col[0] = (int*)malloc(sizeof(int) * n);
    memset(res->col[0], 0, sizeof(int) * n);
    res->col[1] = (int*)malloc(sizeof(int) * n);
    memset(res->col[1], 0, sizeof(int) * n);
    res->n = n;
    res->diagonalLeft[0] = res->diagonalLeft[1] = 0;
    res->diagonalRight[0] = res->diagonalRight[1] = 0;
    return res;
}

/** Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins. */
int ticTacToeMove(TicTacToe* obj, int row, int col, int player) {
    if (!obj) {
        return;
    }
    obj->col[player - 1][col]++;
    obj->row[player - 1][row]++;

    if (row == col) {
        obj->diagonalLeft[player - 1]++;
    }
    if ((row + col) == (obj->n - 1)) {
        obj->diagonalRight[player - 1]++;
    }
    
    if (obj->col[player - 1][col] == obj->n || obj->row[player - 1][row] == obj->n ||
        obj->diagonalLeft[player - 1] == obj->n || obj->diagonalRight[player - 1] == obj->n) {
        return player;
    }
    return 0;
}

void ticTacToeFree(TicTacToe* obj) {
    if (obj) {
        free(obj->col[0]);
        free(obj->col[1]);
        free(obj->row[0]);
        free(obj->row[1]);
    }
}

/**
 * Your TicTacToe struct will be instantiated and called as such:
 * TicTacToe* obj = ticTacToeCreate(n);
 * int param_1 = ticTacToeMove(obj, row, col, player);

 * ticTacToeFree(obj);
*/

------------------------------------------------------------------------------------

351	安卓系统手势解锁
我们都知道安卓有个手势解锁的界面，是一个 3 x 3 的点所绘制出来的网格。

给你两个整数，分别为 ​​m 和 n，其中 1 ≤ m ≤ n ≤ 9，那么请你统计一下有多少种解锁手势，是至少需要经过 m 个点，但是最多经过不超过 n 个点的。

先来了解下什么是一个有效的安卓解锁手势:

每一个解锁手势必须至少经过 m 个点、最多经过 n 个点。
解锁手势里不能设置经过重复的点。
假如手势中有两个点是顺序经过的，那么这两个点的手势轨迹之间是绝对不能跨过任何未被经过的点。
经过点的顺序不同则表示为不同的解锁手势。

解释:

| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | 9 |
无效手势：4 - 1 - 3 - 6 
连接点 1 和点 3 时经过了未被连接过的 2 号点。

无效手势：4 - 1 - 9 - 2
连接点 1 和点 9 时经过了未被连接过的 5 号点。

有效手势：2 - 4 - 1 - 3 - 6
连接点 1 和点 3 是有效的，因为虽然它经过了点 2 ，但是点 2 在该手势中之前已经被连过了。

有效手势：6 - 5 - 4 - 1 - 9 - 2
连接点 1 和点 9 是有效的，因为虽然它经过了按键 5 ，但是点 5 在该手势中之前已经被连过了。

示例:

输入: m = 1，n = 1
输出: 9

class Solution {
public:
    int numberOfPatterns(int m, int n) {
        int res = 0;
        vector<bool> used(10, false);
        vector<vector<int>> neighbor(10, vector<int>(10, 0));
        neighbor[1][3] = neighbor[3][1] = 2;
        neighbor[4][6] = neighbor[6][4] = 5;
        neighbor[7][9] = neighbor[9][7] = 8;
        neighbor[1][7] = neighbor[7][1] = 4;
        neighbor[2][8] = neighbor[8][2] = 5;
        neighbor[3][9] = neighbor[9][3] = 6;
        neighbor[1][9] = neighbor[9][1] = neighbor[3][7] = neighbor[7][3] = 5;
        res += helper(1, 1, 0, m, n, neighbor, used) * 4;
        res += helper(2, 1, 0, m, n, neighbor, used) * 4;
        res += helper(5, 1, 0, m, n, neighbor, used);
        return res;
    }
    int helper(int num, int len, int res, int m, int n, vector<vector<int>>& neighbor, vector<bool>& used)
    {
        if (len >= m) {
            ++res;
        }
        ++len;
        if (len > n) {
            return res;
        }
        used[num] = true;
        for (int next = 1; next <= 9; ++next) {
            int jump = neighbor[num][next];
            if (!used[next] && (jump == 0 || used[jump])) {
                res = helper(next, len, res, m, n, neighbor, used);
            }
        }
        used[num] = false;
        return res;
    }
};

------------------------------------------------------------------------------------

353. 贪吃蛇
请你设计一个 贪吃蛇游戏，该游戏将会在一个 屏幕尺寸 = 宽度 x 高度 的屏幕上运行。如果你不熟悉这个游戏，可以 点击这里 在线试玩。

起初时，蛇在左上角的 (0, 0) 位置，身体长度为 1 个单位。

你将会被给出一个 (行, 列) 形式的食物位置序列。当蛇吃到食物时，身子的长度会增加 1 个单位，得分也会 +1。

食物不会同时出现，会按列表的顺序逐一显示在屏幕上。比方讲，第一个食物被蛇吃掉后，第二个食物才会出现。

当一个食物在屏幕上出现时，它被保证不能出现在被蛇身体占据的格子里。

对于每个 move() 操作，你需要返回当前得分或 -1（表示蛇与自己身体或墙相撞，意味游戏结束）。

示例：

    给定 width = 3, height = 2, 食物序列为 food = [[1,2],[0,1]]。

    Snake snake = new Snake(width, height, food);

    初始时，蛇的位置在 (0,0) 且第一个食物在 (1,2)。

    |S| | |
    | | |F|

    snake.move("R"); -> 函数返回 0

    | |S| |
    | | |F|

    snake.move("D"); -> 函数返回 0

    | | | |
    | |S|F|

    snake.move("R"); -> 函数返回 1 (蛇吃掉了第一个食物，同时第二个食物出现在位置 (0,1))

    | |F| |
    | |S|S|

    snake.move("U"); -> 函数返回 1

    | |F|S|
    | | |S|

    snake.move("L"); -> 函数返回 2 (蛇吃掉了第二个食物)

    | |S|S|
    | | |S|

    snake.move("U"); -> 函数返回 -1 (蛇与边界相撞，游戏结束)


typedef struct node{
    int x;
    int y;
    struct node* next;
} Node;

typedef struct {
    int width;
    int height;
    int** food;
    int foodSize;
    int foodIndex;
    Node* body;  // 第二个元素是头
    int score;
} SnakeGame;

/** Initialize your data structure here.
        @param width - screen width
        @param height - screen height
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0]. */

SnakeGame* snakeGameCreate(int width, int height, int** food, int foodSize, int* foodColSize) {
    SnakeGame* res = (SnakeGame*)malloc(sizeof(SnakeGame));
    res->width = width;
    res->height = height;
    res->score = 0;
    res->food = food;
    res->foodSize = foodSize;
    res->foodIndex = 0;
    res->body = (Node*)malloc(sizeof(Node));
    res->body->next = (Node*)malloc(sizeof(Node));
    res->body->next->x = res->body->next->y = 0;
    res->body->next->next = NULL;
    return res;
}

/** Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
        @return The game's score after the move. Return -1 if game over.
        Game over when snake crosses the screen boundary or bites its body. */
int snakeGameMove(SnakeGame* obj, char * direction) {
    int x = obj->body->next->x, y = obj->body->next->y;
    x = *direction == 'U' ? x - 1 : x;
    x = *direction == 'D' ? x + 1 : x;
    y = *direction == 'L' ? y - 1 : y;
    y = *direction == 'R' ? y + 1 : y;
    //printf("%d %d  ", x, y);

    // 撞到墙
    if (x < 0 || x >= obj->height || y < 0 || y >= obj->width) {
        return -1;
    }

    // 撞到自己    
    Node* bodyTemp = obj->body->next;
    while (bodyTemp != NULL && bodyTemp->next != NULL) {
        if (x == bodyTemp->x && y == bodyTemp->y) {
            return -1;
        }
        bodyTemp = bodyTemp->next;
    }

    Node* temp = (Node*)malloc(sizeof(Node));
    temp->x = x;
    temp->y = y;
    temp->next = obj->body->next;
    obj->body->next = temp;
    
    if (obj->foodIndex < obj->foodSize && *(*(obj->food + obj->foodIndex)) == x && *(*(obj->food + obj->foodIndex) + 1) == y) {
        obj->foodIndex++;
        obj->score++;
        return obj->score;
    }

    // 去最后一个
    while (temp->next != NULL && temp->next->next != NULL) {
        temp = temp->next;
    }
    free(temp->next);
    temp->next = NULL;
    return obj->score;
}

void snakeGameFree(SnakeGame* obj) {
    Node* temp = obj->body;
    while (temp != NULL) {
        Node* del = temp;
        temp = temp->next;
        free(del);
    }
    free(obj);
}

/**
 * Your SnakeGame struct will be instantiated and called as such:
 * SnakeGame* obj = snakeGameCreate(width, height, food, foodSize, foodColSize);
 * int param_1 = snakeGameMove(obj, direction);

 * snakeGameFree(obj);
*/

------------------------------------------------------------------------------------

356. 直线镜像
在一个二维平面空间中，给你 n 个点的坐标。问，是否能找出一条平行于 y 轴的直线，让这些点关于这条直线成镜像排布？

示例 1：

    输入: [[1,1],[-1,1]]
    输出: true
    示例 2：

    输入: [[1,1],[-1,-1]]
    输出: false
    拓展：
    你能找到比 O(n2) 更优的解法吗?

bool isReflected(int** points, int pointsSize, int* pointsColSize){
    //找到中间线 
    int minX = INT_MAX, maxX = INT_MIN;
    for (int i = 0; i < pointsSize; i++) {
        minX = *(*(points + i)) < minX ? *(*(points + i)) : minX;
        maxX = *(*(points + i)) > maxX ? *(*(points + i)) : maxX;
    }
    double middle = minX + (double)(maxX - minX) / 2;

    //定义used
    int *used = (int*)malloc(sizeof(int) * pointsSize);
    memset(used, 0, sizeof(int) * pointsSize);

    //n2遍历
    for (int i = 0; i < pointsSize; i++) {
        if (used[i] == 1 || *(*(points + i)) == middle) {
            continue;
        }
        int flag = 0;
        int j = i + 1;
        for (; j < pointsSize; j++) {
            if (*(*(points + j) + 1) == *(*(points + i) + 1) &&
                *(*(points + j)) == 2 * middle - *(*(points + i))) {
                used[j] = 1;
                flag = 1;
            }
        }
        if (flag == 0) {
            return false;
        }
    }
    free(used);
    return true;
}

------------------------------------------------------------------------------------

360. 有序转化数组
给你一个已经 排好序 的整数数组 nums 和整数 a、b、c。对于数组中的每一个数 x，计算函数值 f(x) = ax2 + bx + c，请将函数值产生的数组返回。

要注意，返回的这个数组必须按照 升序排列，并且我们所期望的解法时间复杂度为 O(n)。

示例 1：

    输入: nums = [-4,-2,2,4], a = 1, b = 3, c = 5
    输出: [3,9,15,33]
    示例 2：

    输入: nums = [-4,-2,2,4], a = -1, b = 3, c = 5
    输出: [-23,-5,1,7]

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* sortTransformedArray(int* nums, int numsSize, int a, int b, int c, int* returnSize) {
    int *res = (int*)malloc(sizeof(int) * numsSize);
    int i = 0, j = numsSize - 1, k;
    if (a >= 0) {
        k = numsSize - 1;
        while (k >= 0) {
            int m = cal(a, b, c, nums[i]);
            int n = cal(a, b, c, nums[j]);
            res[k--] = m > n ? m : n;
            m > n ? i++ : j--;
        }
    }
    else {
        k = 0;
        while (k < numsSize) {
            int m = cal(a, b, c, nums[i]);
            int n = cal(a, b, c, nums[j]);
            res[k++] = m < n ? m : n;
            m < n ? i++: j--;
        }
    }
    *returnSize = numsSize;
    return res;
}

int cal(int a, int b, int c, int num) {
    return a * num * num + b * num + c;
}

------------------------------------------------------------------------------------

361. 轰炸敌人
想象一下炸弹人游戏，在你面前有一个二维的网格来表示地图，网格中的格子分别被以下三种符号占据：

'W' 表示一堵墙
'E' 表示一个敌人
'0'（数字 0）表示一个空位

请你计算一个炸弹最多能炸多少敌人。

由于炸弹的威力不足以穿透墙体，炸弹只能炸到同一行和同一列没被墙体挡住的敌人。

注意：你只能把炸弹放在一个空的格子里

示例:

    输入: [["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]
    输出: 3 
    解释: 对于如下网格

    0 E 0 0 
    E 0 W E 
    0 E 0 0

    假如在位置 (1,1) 放置炸弹的话，可以炸到 3 个敌人

int maxKilledEnemies(char** grid, int gridSize, int* gridColSize) {
    if (!grid || gridSize <= 0 || gridColSize[0] <= 0) {
        return 0;
    }
    int *killed = (int*)malloc(sizeof(int) * gridSize * gridColSize[0]);
    memset(killed, 0, sizeof(int) * gridSize * gridColSize[0]);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridColSize[0]; j++) {
            if (grid[i][j] != 'E') {
                continue;
            }
            int temp;
            for (int k = j - 1; k >= 0; k--) {  // 朝左
                if (grid[i][k] == 'W') {
                    break;
                }
                temp = killed[i * gridColSize[0] + k];
                killed[i * gridColSize[0] + k] = grid[i][k] == '0' ? temp + 1 : temp;
            }
            for (int k = j + 1; k < gridColSize[0]; k++) {  // 朝右
                if (grid[i][k] == 'W') {
                    break;
                }
                temp = killed[i * gridColSize[0] + k];
                killed[i * gridColSize[0] + k] = grid[i][k] == '0' ? temp + 1 : temp;
            }
            for (int k = i - 1; k >= 0; k--) {  // 朝上
                if (grid[k][j] == 'W') {
                    break;
                }
                temp = killed[k * gridColSize[0] + j];
                killed[k * gridColSize[0] + j] = grid[k][j] == '0' ? temp + 1 : temp;
            }
            for (int k = i + 1; k < gridSize; k++) {  // 朝下
                if (grid[k][j] == 'W') {
                    break;
                }
                temp = killed[k * gridColSize[0] + j];
                killed[k * gridColSize[0] + j] = grid[k][j] == '0' ? temp + 1 : temp;
            }
        }
    }
    int res = 0;
    for (int i = 0; i < gridSize * gridColSize[0]; i++) {
        res = killed[i] > res ? killed[i] : res;
    }
    free(killed);
    return res;
}

------------------------------------------------------------------------------------

362. 敲击计数器
设计一个敲击计数器，使它可以统计在过去5分钟内被敲击次数。

每个函数会接收一个时间戳参数（以秒为单位），你可以假设最早的时间戳从1开始，且都是按照时间顺序对系统进行调用（即时间戳是单调递增）。

在同一时刻有可能会有多次敲击。

示例:

HitCounter counter = new HitCounter();

// 在时刻 1 敲击一次。
counter.hit(1);

// 在时刻 2 敲击一次。
counter.hit(2);

// 在时刻 3 敲击一次。
counter.hit(3);

// 在时刻 4 统计过去 5 分钟内的敲击次数, 函数返回 3 。
counter.getHits(4);

// 在时刻 300 敲击一次。
counter.hit(300);

// 在时刻 300 统计过去 5 分钟内的敲击次数，函数返回 4 。
counter.getHits(300);

// 在时刻 301 统计过去 5 分钟内的敲击次数，函数返回 3 。
counter.getHits(301); 
进阶:

如果每秒的敲击次数是一个很大的数字，你的计数器可以应对吗？

typedef struct {
    int time[300];
    int count[300];
} HitCounter;

/** Initialize your data structure here. */

HitCounter* hitCounterCreate() {
    HitCounter *res = (HitCounter*)malloc(sizeof(HitCounter));
    memset(res, 0, sizeof(HitCounter));
    return res;
}

/** Record a hit.
        @param timestamp - The current timestamp (in seconds granularity). */
void hitCounterHit(HitCounter* obj, int timestamp) {
    if (obj->time[timestamp % 300] != timestamp) {
        obj->time[timestamp % 300] = timestamp;
        obj->count[timestamp % 300] = 1;
    }
    else {
        obj->count[timestamp % 300]++;
    }
}

/** Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity). */
int hitCounterGetHits(HitCounter* obj, int timestamp) {
    int res = 0;
    for (int i = 0; i < 300; i++) {
        res += timestamp - obj->time[i] < 300 ? obj->count[i] : 0;
    }
    return res;
}

void hitCounterFree(HitCounter* obj) {
    free(obj);
}

/**
 * Your HitCounter struct will be instantiated and called as such:
 * HitCounter* obj = hitCounterCreate();
 * hitCounterHit(obj, timestamp);

 * int param_2 = hitCounterGetHits(obj, timestamp);

 * hitCounterFree(obj);
*/

------------------------------------------------------------------------------------

364. 加权嵌套序列和 II
给一个嵌套整数序列，请你返回每个数字在序列中的加权和，它们的权重由它们的深度决定。

序列中的每一个元素要么是一个整数，要么是一个序列（这个序列中的每个元素也同样是整数或序列）。

与 前一个问题 不同的是，前一题的权重按照从根到叶逐一增加，而本题的权重从叶到根逐一增加。

也就是说，在本题中，叶子的权重为1，而根拥有最大的权重。

示例 1:

输入: [[1,1],2,[1,1]]
输出: 8 
解释: 四个 1 在深度为 1 的位置， 一个 2 在深度为 2 的位置。
示例 2:

输入: [1,[4,[6]]]
输出: 17 
解释: 一个 1 在深度为 3 的位置， 一个 4 在深度为 2 的位置，一个 6 在深度为 1 的位置。 1*3 + 4*2 + 6*1 = 17。

/**
 * *********************************************************************
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * *********************************************************************
 *
 * // Initializes an empty nested list and return a reference to the nested integer.
 * struct NestedInteger *NestedIntegerInit();
 *
 * // Return true if this NestedInteger holds a single integer, rather than a nested list.
 * bool NestedIntegerIsInteger(struct NestedInteger *);
 *
 * // Return the single integer that this NestedInteger holds, if it holds a single integer
 * // The result is undefined if this NestedInteger holds a nested list
 * int NestedIntegerGetInteger(struct NestedInteger *);
 *
 * // Set this NestedInteger to hold a single integer.
 * void NestedIntegerSetInteger(struct NestedInteger *ni, int value);
 *
 * // Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
 * void NestedIntegerAdd(struct NestedInteger *ni, struct NestedInteger *elem);
 *
 * // Return the nested list that this NestedInteger holds, if it holds a nested list
 * // The result is undefined if this NestedInteger holds a single integer
 * struct NestedInteger **NestedIntegerGetList(struct NestedInteger *);
 *
 * // Return the nested list's size that this NestedInteger holds, if it holds a nested list
 * // The result is undefined if this NestedInteger holds a single integer
 * int NestedIntegerGetListSize(struct NestedInteger *);
 * };
 */

void getDepth(struct NestedInteger** nestedList, int nestedListSize, int depth, int *maxDepth) {
    for (int i = 0; i < nestedListSize; i++) {
        if (!NestedIntegerIsInteger(nestedList[i])) {
            *maxDepth = *maxDepth < depth + 1 ? depth + 1 : *maxDepth;
            getDepth(NestedIntegerGetList(nestedList[i]), NestedIntegerGetListSize(nestedList[i]), depth + 1, maxDepth);
        }
    }
}

void getNum(struct NestedInteger** nestedList, int nestedListSize, int depth, int *nums) {
    for (int i = 0; i < nestedListSize; i++) {
        if (NestedIntegerIsInteger(nestedList[i])) {
            nums[depth] += NestedIntegerGetInteger(nestedList[i]);
        }
        else {
            getNum(NestedIntegerGetList(nestedList[i]), NestedIntegerGetListSize(nestedList[i]), depth + 1, nums);
        }
    }
}

int depthSumInverse(struct NestedInteger** nestedList, int nestedListSize) {
    int maxDepth = 1;
    getDepth(nestedList, nestedListSize, 1, &maxDepth);
    int *nums = (int*)malloc(sizeof(int) * maxDepth);
    memset(nums, 0, sizeof(int) * maxDepth);
    getNum(nestedList, nestedListSize, 0, nums);

    int res = 0;
    for (int i = 0; i < maxDepth; i++) {
        res += nums[i] * (maxDepth - i);
    }
    free(nums);
    return res;
}

------------------------------------------------------------------------------------

366. 寻找完全二叉树的叶子节点
给你一棵完全二叉树，请按以下要求的顺序收集它的全部节点：

依次从左到右，每次收集并删除所有的叶子节点
重复如上过程直到整棵树为空
示例:

输入: [1,2,3,4,5]
  
          1
         / \
        2   3
       / \     
      4   5    

输出: [[4,5,3],[2],[1]]
 

解释:

1. 删除叶子节点 [4,5,3] ，得到如下树结构：

          1
         / 
        2          
 

2. 现在删去叶子节点 [2] ，得到如下树结构：

          1          
 

3. 现在删去叶子节点 [1] ，得到空树：

          []         

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */


/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
int** findLeaves(struct TreeNode* root, int* returnSize, int** returnColumnSizes) {
    int depth = getDepth(root);
    *returnSize = depth;
    int **res = (int**)malloc(sizeof(int*) * depth);
    *returnColumnSizes = (int*)malloc(sizeof(int) * depth);
    memset(*returnColumnSizes, 0, sizeof(int) * depth);
    getColumns(root, depth, res, *returnColumnSizes);

    for (int i = 0; i < depth; i++) {
        res[i] = (int*)malloc(sizeof(int) * (*returnColumnSizes)[i]);
    }

    int *columnIndex = (int*)malloc(sizeof(int) * depth);
    memset(columnIndex, 0, sizeof(int) * depth);
    fillRes(root, depth, res, columnIndex, *returnColumnSizes);
    return res;
}

int getDepth(struct TreeNode* root) {
    if (root == NULL) {
        return 0;
    }
    int left = getDepth(root->left);
    int right = getDepth(root->right);
    return left > right ? left + 1 : right + 1;
}

int getColumns(struct TreeNode* root, int depth, int **res, int* returnColumnSizes) {
    if (root == NULL) {
        return 0;
    }
    int left = getColumns(root->left, depth, res, returnColumnSizes);
    int right = getColumns(root->right, depth, res, returnColumnSizes);
    int level = left > right ? left : right;
    if (level < depth) {
        returnColumnSizes[level]++;
    }
    return level + 1;
}

int fillRes(struct TreeNode* root, int depth, int **res, int *columnIndex, int* returnColumnSizes) {
    if (root == NULL) {
        return 0;
    }
    int left = fillRes(root->left, depth, res, columnIndex, returnColumnSizes);
    int right = fillRes(root->right, depth, res, columnIndex, returnColumnSizes);
    int level = left > right ? left : right;
    if (level < depth) {
        res[level][columnIndex[level]] = root->val;
        columnIndex[level]++;
    }
    return level + 1;
}

------------------------------------------------------------------------------------

369. 给单链表加一
用一个 非空 单链表来表示一个非负整数，然后将这个整数加一。

你可以假设这个整数除了 0 本身，没有任何前导的 0。

这个整数的各个数位按照 高位在链表头部、低位在链表尾部 的顺序排列。

示例:

输入: [1,2,3]
输出: [1,2,4]

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */

struct ListNode* plusOne(struct ListNode* head) {
    if (!head) {
        return head;
    }
    if (helpPlus(head) == 1) {
        struct ListNode* temp = (struct ListNode*)malloc(sizeof(struct ListNode));
        temp->val = 1;
        temp->next = head;
        return temp;
    }
    return head;
}

int helpPlus(struct ListNode* head) {
    if (head == NULL) {
        return 1;
    }
    if (helpPlus(head->next) == 1) {
        if (head->val == 9) {
            head->val = 0;
            return 1;
        }
        head->val++;
    }
    return 0;
}

------------------------------------------------------------------------------------

370. 区间加法
假设你有一个长度为 n 的数组，初始情况下所有的数字均为 0，你将会被给出 k​​​​​​​ 个更新的操作。

其中，每个操作会被表示为一个三元组：[startIndex, endIndex, inc]，你需要将子数组 A[startIndex ... endIndex]（包括 startIndex 和 endIndex）增加 inc。

请你返回 k 次操作后的数组。

示例:

输入: length = 5, updates = [[1,3,2],[2,4,3],[0,2,-2]]
输出: [-2,0,3,5,3]
解释:

初始状态:
[0,0,0,0,0]

进行了操作 [1,3,2] 后的状态:
[0,2,2,2,0]

进行了操作 [2,4,3] 后的状态:
[0,2,5,5,3]

进行了操作 [0,2,-2] 后的状态:
[-2,0,3,5,3]

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* getModifiedArray(int length, int** updates, int updatesSize, int* updatesColSize, int* returnSize) {
    if (length <= 0) {
        return NULL;
    }
    *returnSize = length;
    int* res = (int*)malloc(sizeof(int) * (length + 1));
    memset(res, 0, sizeof(int) * (length + 1));

    for (int i = 0; i < updatesSize; i++) {
        res[updates[i][0]] += updates[i][2];
        res[updates[i][1] + 1] -= updates[i][2];
    }
    for (int i = 1; i < length; i++) {
        res[i] += res[i - 1];
    }
    return res;
}

------------------------------------------------------------------------------------

379. 电话目录管理系统
设计一个电话目录管理系统，让它支持以下功能：

get: 分配给用户一个未被使用的电话号码，获取失败请返回 -1
check: 检查指定的电话号码是否被使用
release: 释放掉一个电话号码，使其能够重新被分配
示例:

// 初始化电话目录，包括 3 个电话号码：0，1 和 2。
PhoneDirectory directory = new PhoneDirectory(3);

// 可以返回任意未分配的号码，这里我们假设它返回 0。
directory.get();

// 假设，函数返回 1。
directory.get();

// 号码 2 未分配，所以返回为 true。
directory.check(2);

// 返回 2，分配后，只剩一个号码未被分配。
directory.get();

// 此时，号码 2 已经被分配，所以返回 false。
directory.check(2);

// 释放号码 2，将该号码变回未分配状态。
directory.release(2);

// 号码 2 现在是未分配状态，所以返回 true。
directory.check(2);

typedef struct {
    int* used;
    int maxNum;
} PhoneDirectory;

/** Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory. */

PhoneDirectory* phoneDirectoryCreate(int maxNumbers) {
    PhoneDirectory* res = (PhoneDirectory*)malloc(sizeof(PhoneDirectory));
    res->used = (int*)malloc(sizeof(int) * maxNumbers);
    memset(res->used, 0, sizeof(int) * maxNumbers);
    res->maxNum = maxNumbers;
    return res;
}

/** Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available. */
int phoneDirectoryGet(PhoneDirectory* obj) {
    for (int i = 0; i < obj->maxNum; i++) {
        if (obj->used[i] == 0) {
            obj->used[i] = 1;
            return i;
        }
    }
    return -1;
}

/** Check if a number is available or not. */
bool phoneDirectoryCheck(PhoneDirectory* obj, int number) {
    if (number >= 0 && number < obj->maxNum && obj->used[number] == 0) {
        return true;
    }
    return false;
}

/** Recycle or release a number. */
void phoneDirectoryRelease(PhoneDirectory* obj, int number) {
    if (number >= 0 && number < obj->maxNum) {
        obj->used[number] = 0;
    }
}

void phoneDirectoryFree(PhoneDirectory* obj) {
    free(obj->used);
    free(obj);
}

/**
 * Your PhoneDirectory struct will be instantiated and called as such:
 * PhoneDirectory* obj = phoneDirectoryCreate(maxNumbers);
 * int param_1 = phoneDirectoryGet(obj);
 
 * bool param_2 = phoneDirectoryCheck(obj, number);
 
 * phoneDirectoryRelease(obj, number);
 
 * phoneDirectoryFree(obj);
*/

------------------------------------------------------------------------------------

418. 屏幕可显示句子的数量
给你一个 rows x cols 的屏幕和一个用 非空 的单词列表组成的句子，请你计算出给定句子可以在屏幕上完整显示的次数。

注意：

一个单词不能拆分成两行。
单词在句子中的顺序必须保持不变。
在一行中 的两个连续单词必须用一个空格符分隔。
句子中的单词总量不会超过 100。
每个单词的长度大于 0 且不会超过 10。
1 ≤ rows, cols ≤ 20,000.

示例 1：

输入：
rows = 2, cols = 8, 句子 sentence = ["hello", "world"]

输出：
1

解释：
hello---
world---

字符 '-' 表示屏幕上的一个空白位置。

示例 2：

输入：
rows = 3, cols = 6, 句子 sentence = ["a", "bcd", "e"]

输出：
2

解释：
a-bcd- 
e-a---
bcd-e-

字符 '-' 表示屏幕上的一个空白位置。

示例 3：

输入：
rows = 4, cols = 5, 句子 sentence = ["I", "had", "apple", "pie"]

输出：
1

解释：
I-had
apple
pie-I
had--

字符 '-' 表示屏幕上的一个空白位置。

int wordsTyping(char ** sentence, int sentenceSize, int rows, int cols){
    int res = 0;
    int wordIndex = 0;
    int sentenceLen = 0;
    for (int i = 0; i < sentenceSize; i++) {
        sentenceLen += strlen(sentence[i]) + 1;
    }
    for (int i = 0; i < rows; i++) {
        int rowPos = 0;
        while (cols - rowPos >= (int)strlen(sentence[wordIndex])) {  // 能放下
            int multiAll = (cols - rowPos) / sentenceLen;
            if (multiAll > 0) {
                res += multiAll;
                rowPos += multiAll * sentenceLen;
            }
            if (cols - rowPos >= (int)strlen(sentence[wordIndex])) {
                rowPos += strlen(sentence[wordIndex]) + 1;
                wordIndex++;
                if (wordIndex == sentenceSize) {
                    res++;
                    wordIndex = 0;
                }
            }
        }
    }
    return res;
}

------------------------------------------------------------------------------------

426. 将二叉搜索树转化为排序的双向链表
将一个二叉搜索树就地转化为一个已排序的双向循环链表。可以将左右孩子指针作为双向循环链表的前驱和后继指针。

为了让您更好地理解问题，以下面的二叉搜索树为例：
 
我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

下图显示了转化后的二叉搜索树，实线表示后继关系，虚线表示前驱关系。

Node* treeToDoublyList(Node* root) {
    if (root == NULL) {
        return NULL;
    }
    pair<Node*, Node*> res = help(root);
    res.first->left = res.second;
    res.second->right = res.first;
    return res.first;
}

pair<Node*, Node*> help(Node* root) {
    if (root->left == NULL && root->right == NULL) {
        return {root, root};
    }
    if (root->right == NULL) {
        pair<Node*, Node*> l = help(root->left);
        root->left = l.second;
        l.second->right = root;
        return {l.first, root};
    }
    if (root->left == NULL) {
        pair<Node*, Node*> r = help(root->right);
        root->right = r.first;
        r.first->left = root;
        return {root, r.second};
    }
    pair<Node*, Node*> l = help(root->left);
    root->left = l.second;
    l.second->right = root; 
    pair<Node*, Node*> r = help(root->right);        
    root->right = r.first;
    r.first->left = root;
    return {l.first,r.second};
}

------------------------------------------------------------------------------------

439. 三元表达式解析器
给定一个以字符串表示的任意嵌套的三元表达式，计算表达式的值。你可以假定给定的表达式始终都是有效的并且只包含数字 0-9, ?, :, T 和 F (T 和 F 分别表示真和假）。

注意：

给定的字符串长度 ≤ 10000。
所包含的数字都只有一位数。
条件表达式从右至左结合（和大多数程序设计语言类似）。
条件是 T 和 F其一，即条件永远不会是数字。
表达式的结果是数字 0-9, T 或者 F。
 

示例 1：

输入： "T?2:3"

输出： "2"

解释： 如果条件为真，结果为 2；否则，结果为 3。
 
示例 2：

输入： "F?1:T?4:5"

输出： "4"

解释： 条件表达式自右向左结合。使用括号的话，相当于：

             "(F ? 1 : (T ? 4 : 5))"                   "(F ? 1 : (T ? 4 : 5))"
          -> "(F ? 1 : 4)"                 或者     -> "(T ? 4 : 5)"
          -> "4"                                    -> "4"

示例 3：

输入： "T?T?F:5:3"

输出： "F"

解释： 条件表达式自右向左结合。使用括号的话，相当于：

             "(T ? (T ? F : 5) : 3)"                   "(T ? (T ? F : 5) : 3)"
          -> "(T ? F : 3)"                 或者       -> "(T ? F : 5)"
          -> "F"                                     -> "F"

typedef struct stack {
    char data[10000];
    int top;
} Stack;

int push(Stack *s, char value) {
    if (s->top == 10000 - 1) {
        return -1;
    }
    s->top++;
    s->data[s->top] = value;
    return 0;
}

int pop(Stack *s, char *value) {
    if (s->top == -1) {
        return -1;
    }
    *value = s->data[s->top];
    s->top--;
    return 0;
}

int top(Stack *s) {
    if (s->top == -1) {
        return 0;
    }
    return s->data[s->top];
}

char * parseTernary(char * expression) {
    if (expression == NULL) {
        return NULL;
    }
    char *res = (char*)malloc(sizeof(char) * 2);
    memset(res, 0, sizeof(char) * 2); 
    Stack *stk = (Stack*)malloc(sizeof(Stack));
    stk->top = -1;
    
    char a, b, c;
    for (int i = strlen(expression) - 1; i >= 0; i--) {
        if (top(stk) == '?') {
            a = expression[i];
            pop(stk, &b);
            pop(stk, &b);
            pop(stk, &c);
            pop(stk, &c);
            push(stk, a == 'T' ? b : c);
        }
        else {
            push(stk, expression[i]);
        }
    }
    pop(stk, res);
    free(stk);
    return res;
}

------------------------------------------------------------------------------------

444. 序列重建
验证原始的序列 org 是否可以从序列集 seqs 中唯一地重建。序列 org 是 1 到 n 整数的排列，其中 1 ≤ n ≤ 104。重建是指在序列集 seqs 中构建最短的公共超序列。（即使得所有  seqs 中的序列都是该最短序列的子序列）。确定是否只可以从 seqs 重建唯一的序列，且该序列就是 org 。

示例 1：

输入：
org: [1,2,3], seqs: [[1,2],[1,3]]

输出：
false

解释：
[1,2,3] 不是可以被重建的唯一的序列，因为 [1,3,2] 也是一个合法的序列。
 

示例 2：

输入：
org: [1,2,3], seqs: [[1,2]]

输出：
false

解释：
可以重建的序列只有 [1,2]。
 

示例 3：

输入：
org: [1,2,3], seqs: [[1,2],[1,3],[2,3]]

输出：
true

解释：
序列 [1,2], [1,3] 和 [2,3] 可以被唯一地重建为原始的序列 [1,2,3]。
 

示例 4：

输入：
org: [4,1,5,2,6,3], seqs: [[5,2,6,3],[4,1,5,2]]

输出：
true

bool sequenceReconstruction(int* org, int orgSize, int** seqs, int seqsSize, int* seqsColSize){
    // 双方元素相对位置一致：  遍历seqs，每个元素在org中的位置都比后边的要小
    // 双方元素一致：         遍历某一方的元素时，不能超过另一方的范围；并且能够让对方消耗完
    // 唯一：                seq中的每个元素与后一个元素，在org中也是挨着时，才算是有效
    if (org == NULL || orgSize <= 0 || seqsSize <= 0 || seqsColSize == NULL) {
        return false;
    }
    int *pos = (int*)malloc(sizeof(int) * (orgSize + 1));
    char *flag = (char*)malloc(sizeof(char) * (orgSize + 1));
    memset(pos, 0, sizeof(int) * (orgSize + 1));
    memset(flag, 0, sizeof(char) * (orgSize + 1));
    int solveFlag = 0;
    for (int i = 0; i < orgSize; i++) {
        if (org[i] < 1 || org[i] > orgSize) {
            return false;
        }
        pos[org[i]] = i;
    }
    
    int orgRemainCnt = orgSize - 1;
    for (int i = 0; i < seqsSize; i++) {
        for (int j = 0; j < seqsColSize[i]; j++) {
            if(seqs[i][j] < 1 || seqs[i][j] > orgSize || (j < seqsColSize[i] - 1 && 
              (seqs[i][j + 1] < 1 || seqs[i][j + 1] > orgSize || pos[seqs[i][j]] > pos[seqs[i][j + 1]]
               || seqs[i][j] == seqs[i][j + 1]))) {
                return false;
            }
            if (seqs[i][j] >= 1 && seqs[i][j] <= orgSize) {
                solveFlag = 1;
            }
            if (flag[seqs[i][j]] == 0 && j < seqsColSize[i] - 1 && pos[seqs[i][j]] + 1 == pos[seqs[i][j + 1]]) {
                flag[seqs[i][j]] = 1;
                orgRemainCnt--;
                solveFlag = 1;
            }
        }
    }
    return orgRemainCnt == 0 && solveFlag == 1;
}

------------------------------------------------------------------------------------

469. 凸多边形
给定一个按顺序连接的多边形的顶点，判断该多边形是否为凸多边形。（凸多边形的定义）

注：

顶点个数至少为 3 个且不超过 10,000。
坐标范围为 -10,000 到 10,000。
你可以假定给定的点形成的多边形均为简单多边形（简单多边形的定义）。换句话说，保证每个顶点处恰好是两条边的汇合点，并且这些边 互不相交 。
 

示例 1：

[[0,0],[0,1],[1,1],[1,0]]

输出： True

示例 2：

[[0,0],[0,10],[10,10],[10,0],[5,5]]

输出： False

bool isConvex(int** points, int pointsSize, int* pointsColSize){
    long long cur = 0, pre = 0;
    for (int i = 0; i < pointsSize; i++) {
        int xf = points[(i + 1) % pointsSize][0] - points[i][0];
        int yf = points[(i + 1) % pointsSize][1] - points[i][1];
        int xb = points[(i + 2) % pointsSize][0] - points[i][0];
        int yb = points[(i + 2) % pointsSize][1] - points[i][1];
        cur = xf * yb - yf * xb;
        if (cur != 0) {
            if (cur * pre < 0) {
                return false;
            }
            pre = cur;
        }
    }
    return true;
}

------------------------------------------------------------------------------------

484. 寻找排列
现在给定一个只由字符 'D' 和 'I' 组成的 秘密签名。'D' 表示两个数字间的递减关系，'I' 表示两个数字间的递增关系。并且 秘密签名 是由一个特定的整数数组生成的，该数组唯一地包含 1 到 n 中所有不同的数字（秘密签名的长度加 1 等于 n）。例如，秘密签名 "DI" 可以由数组 [2,1,3] 或 [3,1,2] 生成，但是不能由数组 [3,2,4] 或 [2,1,3,4] 生成，因为它们都不是合法的能代表 "DI" 秘密签名 的特定串。

现在你的任务是找到具有最小字典序的 [1, 2, ... n] 的排列，使其能代表输入的 秘密签名。

示例 1：

输入： "I"
输出： [1,2]
解释： [1,2] 是唯一合法的可以生成秘密签名 "I" 的特定串，数字 1 和 2 构成递增关系。
 
示例 2：

输入： "DI"
输出： [2,1,3]
解释： [2,1,3] 和 [3,1,2] 可以生成秘密签名 "DI"，
但是由于我们要找字典序最小的排列，因此你需要输出 [2,1,3]。
 
注：

输出字符串只会包含字符 'D' 和 'I'。
输入字符串的长度是一个正整数且不会超过 10,000。

int* findPermutation(char * s, int* returnSize) {
    *returnSize = strlen(s) + 1;
    int *res = (int*)malloc(sizeof(int) * *returnSize);    
    for (int i = 0; i < *returnSize; i++) {
        res[i] = i + 1;
    }
    for (int i = 0; i < strlen(s); i++) {
        if (s[i] == 'I') {
            continue;
        }
        int firstD = i;
        while (s[i] == 'D' && i < strlen(s)) {
            i++;
        }
        reversePermutation(res, firstD, i);
    }
    return res;
}

void reversePermutation(int *res, int first, int end) {
    if (res == NULL) {
        return;
    }
    while (first < end) {
        int temp = res[first];
        res[first] = res[end];
        res[end] = temp;
        first++;
        end--;
    }
}

------------------------------------------------------------------------------------

487. 最大连续1的个数 II
给定一个二进制数组，你可以最多将 1 个 0 翻转为 1，找出其中最大连续 1 的个数。

示例 1：

输入：[1,0,1,1,0]
输出：4
解释：翻转第一个 0 可以得到最长的连续 1。
     当翻转以后，最大连续 1 的个数为 4。
 
注：

输入数组只包含 0 和 1.
输入数组的长度为正整数，且不超过 10,000
 
进阶：
如果输入的数字是作为 无限流 逐个输入如何处理？换句话说，内存不能存储下所有从流中输入的数字。您可以有效地解决吗？

int findMaxConsecutiveOnes(int* nums, int numsSize){
    int maxLen = 0, this1Len = 0, pre1Len;
    //如果一直是0就是1，有1加1，第二个0清掉以前的pre1Len
    for (int i = 0; i < numsSize; i++) {
        this1Len++;
        if (nums[i] == 0) {
            pre1Len = this1Len;
            this1Len = 0;
        }
        maxLen = this1Len + pre1Len > maxLen ? this1Len + pre1Len : maxLen;
    }
    return maxLen;
}

------------------------------------------------------------------------------------

490. 迷宫
由空地和墙组成的迷宫中有一个球。球可以向上下左右四个方向滚动，但在遇到墙壁前不会停止滚动。当球停下时，可以选择下一个方向。

给定球的起始位置，目的地和迷宫，判断球能否在目的地停下。

迷宫由一个0和1的二维数组表示。 1表示墙壁，0表示空地。你可以假定迷宫的边缘都是墙壁。起始位置和目的地的坐标通过行号和列号给出。

示例 1:

输入 1: 迷宫由以下二维数组表示

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

输入 2: 起始位置坐标 (rowStart, colStart) = (0, 4)
输入 3: 目的地坐标 (rowDest, colDest) = (4, 4)

输出: true

解析: 一个可能的路径是 : 左 -> 下 -> 左 -> 下 -> 右 -> 下 -> 右。

示例 2:

输入 1: 迷宫由以下二维数组表示

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

输入 2: 起始位置坐标 (rowStart, colStart) = (0, 4)
输入 3: 目的地坐标 (rowDest, colDest) = (3, 2)

输出: false

解析: 没有能够使球停在目的地的路径。

注意:

迷宫中只有一个球和一个目的地。
球和目的地都在空地上，且初始时它们不在同一位置。
给定的迷宫不包括边界 (如图中的红色矩形), 但你可以假设迷宫的边缘都是墙壁。
迷宫至少包括2块空地，行数和列数均不超过100。

bool helpPath(int** maze, int Size, int ColSize, int *dirX, int *dirY, int x, int y, int desX, int desY) {
    if (x == desX && y == desY) {
        return true;
    }
    maze[x][y] = -1;
    bool res = false;
    for (int i = 0; i < 4; i++) {
        int thisX = x, thisY = y;
        while (thisX >= 0 && thisX < Size && thisY >= 0 && thisY < ColSize && maze[thisX][thisY] != 1) {
            thisX += dirX[i];
            thisY += dirY[i];
        }
        thisX -= dirX[i];
        thisY -= dirY[i];
        if (maze[thisX][thisY] != -1) {
            res = res || helpPath(maze, Size, ColSize, dirX, dirY, thisX, thisY, desX, desY);
        }
    }
    return res;
}

bool hasPath(int** maze, int mazeSize, int* mazeColSize, int* start, int startSize, int* destination, int destinationSize) {
    int dirX[4] = {0,  0, -1, 1};  // 左右上下
    int dirY[4] = {-1, 1,  0, 0};
    return helpPath(maze, mazeSize, mazeColSize[0], dirX, dirY, start[0], start[1], destination[0], destination[1]);
}

------------------------------------------------------------------------------------

505. 迷宫 II
由空地和墙组成的迷宫中有一个球。球可以向上下左右四个方向滚动，但在遇到墙壁前不会停止滚动。当球停下时，可以选择下一个方向。

给定球的起始位置，目的地和迷宫，找出让球停在目的地的最短距离。距离的定义是球从起始位置（不包括）到目的地（包括）经过的空地个数。如果球无法停在目的地，返回 -1。

迷宫由一个0和1的二维数组表示。 1表示墙壁，0表示空地。你可以假定迷宫的边缘都是墙壁。起始位置和目的地的坐标通过行号和列号给出。

示例 1:

输入 1: 迷宫由以下二维数组表示

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

输入 2: 起始位置坐标 (rowStart, colStart) = (0, 4)
输入 3: 目的地坐标 (rowDest, colDest) = (4, 4)

输出: 12

解析: 一条最短路径 : left -> down -> left -> down -> right -> down -> right。
             总距离为 1 + 1 + 3 + 1 + 2 + 2 + 2 = 12。

示例 2:

输入 1: 迷宫由以下二维数组表示

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

输入 2: 起始位置坐标 (rowStart, colStart) = (0, 4)
输入 3: 目的地坐标 (rowDest, colDest) = (3, 2)

输出: -1

解析: 没有能够使球停在目的地的路径。

注意:

迷宫中只有一个球和一个目的地。
球和目的地都在空地上，且初始时它们不在同一位置。
给定的迷宫不包括边界 (如图中的红色矩形), 但你可以假设迷宫的边缘都是墙壁。
迷宫至少包括2块空地，行数和列数均不超过100。

void helpDistance(int** maze, int Size, int ColSize, int *used, int *dirX, int *dirY, int x, int y, int desX, int desY) {
    if (x == desX && y == desY) {
        return;
    }
    for (int i = 0; i < 4; i++) {
        int thisX = x, thisY = y, thisres = used[x * ColSize + y];
        while (thisX >= 0 && thisX < Size && thisY >= 0 && thisY < ColSize && maze[thisX][thisY] != 1) {
            thisX += dirX[i];
            thisY += dirY[i];
            thisres++;
        }
        thisX -= dirX[i];
        thisY -= dirY[i];
        thisres--;
        if (thisres < used[thisX * ColSize + thisY]) {
            used[thisX * ColSize + thisY] = thisres;
            helpDistance(maze, Size, ColSize, used, dirX, dirY, thisX, thisY, desX, desY);
        }
    }
}

int shortestDistance(int** maze, int mazeSize, int* mazeColSize, int* start, int startSize, int* destination, int destinationSize) {
    int dirX[4] = { 0,  0, -1, 1 };  // 左右上下
    int dirY[4] = { -1, 1,  0, 0 };
    int *used = (int*)malloc(sizeof(int) * mazeSize * mazeColSize[0]);
    for (int i = 0; i < mazeSize * mazeColSize[0]; i++) {
        used[i] = INT_MAX;
    }
    used[mazeColSize[0] * start[0] + start[1]] = 0;    
    helpDistance(maze, mazeSize, mazeColSize[0], used, dirX, dirY, start[0], start[1], destination[0], destination[1]);
    int res = used[mazeColSize[0] * destination[0] + destination[1]];
    free(used);
    return res == INT_MAX ? -1 : res;
}

------------------------------------------------------------------------------------

510. 二叉搜索树中的中序后继 II
给定一棵二叉搜索树和其中的一个节点，找到该节点在树中的中序后继。

一个结点 p 的中序后继是键值比 p.val大所有的结点中键值最小的那个。

你可以直接访问结点，但无法直接访问树。每个节点都会有其父节点的引用。

示例 1:
输入: 
root = {"$id":"1","left":{"$id":"2","left":null,"parent":{"$ref":"1"},"right":null,"val":1},"parent":null,"right":{"$id":"3","left":null,"parent":{"$ref":"1"},"right":null,"val":3},"val":2}
p = 1
输出: 2
解析: 1的中序后继结点是2。注意p和返回值都是Node类型的。

示例 2:
输入: 
root = {"$id":"1","left":{"$id":"2","left":{"$id":"3","left":{"$id":"4","left":null,"parent":{"$ref":"3"},"right":null,"val":1},"parent":{"$ref":"2"},"right":null,"val":2},"parent":{"$ref":"1"},"right":{"$id":"5","left":null,"parent":{"$ref":"2"},"right":null,"val":4},"val":3},"parent":null,"right":{"$id":"6","left":null,"parent":{"$ref":"1"},"right":null,"val":6},"val":5}
p = 6
输出: null
解析: 该结点没有中序后继，因此返回null。

示例 3:
输入: 
root = {"$id":"1","left":{"$id":"2","left":{"$id":"3","left":{"$id":"4","left":null,"parent":{"$ref":"3"},"right":null,"val":2},"parent":{"$ref":"2"},"right":{"$id":"5","left":null,"parent":{"$ref":"3"},"right":null,"val":4},"val":3},"parent":{"$ref":"1"},"right":{"$id":"6","left":null,"parent":{"$ref":"2"},"right":{"$id":"7","left":{"$id":"8","left":null,"parent":{"$ref":"7"},"right":null,"val":9},"parent":{"$ref":"6"},"right":null,"val":13},"val":7},"val":6},"parent":null,"right":{"$id":"9","left":{"$id":"10","left":null,"parent":{"$ref":"9"},"right":null,"val":17},"parent":{"$ref":"1"},"right":{"$id":"11","left":null,"parent":{"$ref":"9"},"right":null,"val":20},"val":18},"val":15}
p = 15
输出: 17

示例 4:
输入: 
root = {"$id":"1","left":{"$id":"2","left":{"$id":"3","left":{"$id":"4","left":null,"parent":{"$ref":"3"},"right":null,"val":2},"parent":{"$ref":"2"},"right":{"$id":"5","left":null,"parent":{"$ref":"3"},"right":null,"val":4},"val":3},"parent":{"$ref":"1"},"right":{"$id":"6","left":null,"parent":{"$ref":"2"},"right":{"$id":"7","left":{"$id":"8","left":null,"parent":{"$ref":"7"},"right":null,"val":9},"parent":{"$ref":"6"},"right":null,"val":13},"val":7},"val":6},"parent":null,"right":{"$id":"9","left":{"$id":"10","left":null,"parent":{"$ref":"9"},"right":null,"val":17},"parent":{"$ref":"1"},"right":{"$id":"11","left":null,"parent":{"$ref":"9"},"right":null,"val":20},"val":18},"val":15}
p = 13
输出: 15

Node* inorderSuccessor(Node* node) {
    if (!node) {
        return NULL;
    }
    if (node->right) {
        node = node->right;
        while (node->left) {
            node = node->left;
        }
        return node;
    }
    if (node->parent && node->parent->left == node) {
        return node->parent;
    }
    else {
        while (node->parent && node->parent->right == node) {
            node = node->parent;            
        }
        return node->parent;
    }
    return node;
}
    
------------------------------------------------------------------------------------

531. 孤独像素 I
给定一幅黑白像素组成的图像, 计算黑色孤独像素的数量。

图像由一个由‘B’和‘W’组成二维字符数组表示, ‘B’和‘W’分别代表黑色像素和白色像素。

黑色孤独像素指的是在同一行和同一列不存在其他黑色像素的黑色像素。

示例:

输入: 
[['W', 'W', 'B'],
 ['W', 'B', 'W'],
 ['B', 'W', 'W']]

输出: 3
解析: 全部三个'B'都是黑色孤独像素。
 
注意:

输入二维数组行和列的范围是 [1,500]。

int findLonelyPixel(char** picture, int pictureSize, int* pictureColSize){
    int* row = (int*)malloc(sizeof(int) * pictureSize);
    int* col = (int*)malloc(sizeof(int) * pictureColSize[0]);
    
    int cnt = 0, rowCnt = 0, colCnt = 0;
    for (int i = 0; i < pictureSize; i++) {
        cnt = 0;
        for (int j = 0; j < pictureColSize[0]; j++) {
            cnt = picture[i][j] == 'B' ? cnt + 1 : cnt;
        }
        row[i] = cnt == 1 ? 1 : 0;
        rowCnt = row[i] == 1 ? rowCnt + 1 : rowCnt;
    }
    for (int j = 0; j < pictureColSize[0]; j++) {
        cnt = 0;
        for (int i = 0; i < pictureSize; i++) {
            cnt = picture[i][j] == 'B' ? cnt + 1 : cnt;
        }
        col[j] = cnt == 1 ? 1 : 0;
        colCnt = col[j] == 1 ? colCnt + 1 : colCnt;
    }
    free(row);
    free(col);
    return rowCnt < colCnt ? rowCnt : colCnt;
}

------------------------------------------------------------------------------------

533. 孤独像素 II
给定一幅由黑色像素和白色像素组成的图像， 与一个正整数N, 找到位于某行 R 和某列 C 中且符合下列规则的黑色像素的数量:

行R 和列C都恰好包括N个黑色像素。
列C中所有黑色像素所在的行必须和行R完全相同。
图像由一个由‘B’和‘W’组成二维字符数组表示, ‘B’和‘W’分别代表黑色像素和白色像素。

示例:

输入:                                            
[['W', 'B', 'W', 'B', 'B', 'W'],    
 ['W', 'B', 'W', 'B', 'B', 'W'],    
 ['W', 'B', 'W', 'B', 'B', 'W'],    
 ['W', 'W', 'B', 'W', 'B', 'W']] 

N = 3
输出: 6
解析: 所有粗体的'B'都是我们所求的像素(第1列和第3列的所有'B').
        0    1    2    3    4    5         列号                                          
0    [['W', 'B', 'W', 'B', 'B', 'W'],    
1     ['W', 'B', 'W', 'B', 'B', 'W'],    
2     ['W', 'B', 'W', 'B', 'B', 'W'],    
3     ['W', 'W', 'B', 'W', 'B', 'W']]    
行号

以R = 0行和C = 1列的'B'为例:
规则 1，R = 0行和C = 1列都恰好有N = 3个黑色像素. 
规则 2，在C = 1列的黑色像素分别位于0，1和2行。它们都和R = 0行完全相同。

注意:

输入二维数组行和列的范围是 [1,200]。

看不懂 不想做

------------------------------------------------------------------------------------

536. 从字符串生成二叉树
你需要从一个包括括号和整数的字符串构建一棵二叉树。

输入的字符串代表一棵二叉树。它包括整数和随后的0，1或2对括号。整数代表根的值，一对括号内表示同样结构的子树。

若存在左子结点，则从左子结点开始构建。

示例:

输入: "4(2(3)(1))(6(5))"
输出: 返回代表下列二叉树的根节点:

       4
     /   \
    2     6
   / \   / 
  3   1 5   
 

注意:

输入字符串中只包含 '(', ')', '-' 和 '0' ~ '9' 
空树由 "" 而非"()"表示。

typedef struct stack {
    struct TreeNode *node[10000];
    int top;
} Stack;
int push(Stack *s, struct TreeNode *value) {
    if (s->top == 10000 - 1) {
        return -1;
    }
    s->top++;
    s->node[s->top] = value;
    return 0;
}
int pop(Stack *s, struct TreeNode **value) {
    if (s->top == -1) {
        return -1;
    }
    *value = s->node[s->top];
    s->top--;
    return 0;
}
struct TreeNode* top(Stack *s) {
    if (s->top == -1) {
        return 0;
    }
    return s->node[s->top];
}
int getNum(char *s, int *index) {
    int negtive = 0;
    if (s[(*index)] == '-') {
        (*index)++;
        negtive = 1;
    }
    int res = 0;    
    while (s[(*index)] >= '0' && s[(*index)] <= '9') {
        res = s[(*index)] - '0' + res * 10;
        (*index)++;
    }
    return negtive == 0 ? res : 0 - res;
}
struct TreeNode* str2tree(char * s) {
    if (strlen(s) == 0) {
        return NULL;
    }
    int sIndex = 0; 
    struct TreeNode *res = (struct TreeNode*)malloc(sizeof(struct TreeNode));
    res->val = getNum(s, &sIndex);
    res->left = res->right = NULL;    
    Stack *stack = (Stack*)malloc(sizeof(Stack));
    struct TreeNode *father = res;
    while (s[sIndex] != 0) {
        if (s[sIndex] == '(') {
            sIndex++;
            push(stack, father);
            struct TreeNode *temp = (struct TreeNode*)malloc(sizeof(struct TreeNode));
            temp->val = getNum(s, &sIndex);
            temp->left = temp->right = NULL;
            if (father->left == NULL) {
                father->left = temp;
                father = father->left;
            }
            else {
                father->right = temp;
                father = father->right;
            }
        }
        else if (s[sIndex] == ')') {
            pop(stack, &father);
            sIndex++;
        }
    }
    return res;
}

------------------------------------------------------------------------------------

544. 输出比赛匹配对
在 NBA 季后赛中，我们总是安排较强的队伍对战较弱的队伍，例如用排名第 1 的队伍和第 n 的队伍对决，这是一个可以让比赛更加有趣的好策略。现在，给你 n 支队伍，你需要以字符串格式输出它们的 最终 比赛配对。

n 支队伍按从 1 到 n 的正整数格式给出，分别代表它们的初始排名（排名 1 最强，排名 n 最弱）。我们用括号（'(', ')'）和逗号（','）来表示匹配对——括号（'(', ')'）表示匹配，逗号（','）来用于分割。 在每一轮的匹配过程中，你都需要遵循将强队与弱队配对的原则。

 

示例 1：

输入: 2
输出: (1,2)
解析: 
初始地，我们有队1和队2两支队伍，按照1，2排列。
因此 用 '(', ')' 和 ','来将队1和队2进行配对，得到最终答案。
示例 2：

输入: 4
输出: ((1,4),(2,3))
解析: 
在第一轮，我们将队伍1和4配对，2和3配对，以满足将强队和弱队搭配的效果。得到(1,4),(2,3).
在第二轮，(1,4) 和 (2,3) 的赢家需要进行比赛以确定最终赢家，因此需要再在外面加一层括号。
于是最终答案是((1,4),(2,3))。
示例 3：

输入: 8
输出: (((1,8),(4,5)),((2,7),(3,6)))
解析: 
第一轮: (1,8),(2,7),(3,6),(4,5)
第二轮: ((1,8),(4,5)),((2,7),(3,6))
第三轮 (((1,8),(4,5)),((2,7),(3,6)))
由于第三轮会决出最终胜者，故输出答案为(((1,8),(4,5)),((2,7),(3,6)))。

string findContestMatch(int n) {
    vector<string> str(n);
    for (int i = 0; i < n; i++) {
        str[i] = to_string(i + 1);
    }
    while (n > 1) {
        for (int i = 0; i < n / 2; i++) {
            str[i] = '(' + str[i] + ',' + str[n - i - 1] + ')';
        }
        n /= 2;
    }
    return str[0];
}

------------------------------------------------------------------------------------
545. 二叉树的边界
给定一棵二叉树，以逆时针顺序从根开始返回其边界。边界按顺序包括左边界、叶子结点和右边界而不包括重复的结点。 (结点的值可能重复)

左边界的定义是从根到最左侧结点的路径。右边界的定义是从根到最右侧结点的路径。若根没有左子树或右子树，则根自身就是左边界或右边界。注意该定义只对输入的二叉树有效，而对子树无效。

最左侧结点的定义是：在左子树存在时总是优先访问，如果不存在左子树则访问右子树。重复以上操作，首先抵达的结点就是最左侧结点。

最右侧结点的定义方式相同，只是将左替换成右。

示例 1

输入:
  1
   \
    2
   / \
  3   4

输出:
[1, 3, 4, 2]

解析:
根不存在左子树，故根自身即为左边界。
叶子结点是3和4。
右边界是1，2，4。注意逆时针顺序输出需要你输出时调整右边界顺序。
以逆时针顺序无重复地排列边界，得到答案[1,3,4,2]。
 

示例 2

输入:
    ____1_____
   /          \
  2            3
 / \          / 
4   5        6   
   / \      / \
  7   8    9  10  
       
输出:
[1,2,4,7,8,9,10,6,3]

解析:
左边界是结点1,2,4。(根据定义，4是最左侧结点)
叶子结点是结点4,7,8,9,10。
右边界是结点1,3,6,10。(10是最右侧结点)
以逆时针顺序无重复地排列边界，得到答案 [1,2,4,7,8,9,10,6,3]。

vector<int> boundaryOfBinaryTree(TreeNode* root) {
    if (!root) {
        return {};
    }
    vector<int> res, right;
    TreeNode *l = root->left, *r = root->right, *p = root;
    
    if (root->left || root->right) {
        res.push_back(root->val);
    }
    while (l && (l->left || l->right)) {  // 左边节点
        res.push_back(l->val);
        if (l->left) {
            l = l->left;
        }
        else {
            l = l->right;
        }
    }

    stack<TreeNode*> st;
    while (p || !st.empty()) {  // 叶子节点
        if (p) {
            st.push(p);
            if (!p->left && !p->right) {
                res.push_back(p->val);
            }
            p = p->left;
        }
        else {
            p = st.top();
            st.pop();
            p = p->right;
        }
    }

    while (r && (r->left || r->right)) {  // 右边节点
        right.push_back(r->val);
        if (r->right) {
            r = r->right;
        }
        else {
            r = r->left;
        }
    }
    res.insert(res.end(), right.rbegin(), right.rend());
    return res;
}

------------------------------------------------------------------------------------

548. 将数组分割成和相等的子数组
给定一个有 n 个整数的数组，你需要找到满足以下条件的三元组 (i, j, k) ：

0 < i, i + 1 < j, j + 1 < k < n - 1
子数组 (0, i - 1)，(i + 1, j - 1)，(j + 1, k - 1)，(k + 1, n - 1) 的和应该相等。
这里我们定义子数组 (L, R) 表示原数组从索引为L的元素开始至索引为R的元素。 

示例:

输入: [1,2,1,2,1,2,1]
输出: True
解释:
i = 1, j = 3, k = 5. 
sum(0, i - 1) = sum(0, 0) = 1
sum(i + 1, j - 1) = sum(2, 2) = 1
sum(j + 1, k - 1) = sum(4, 4) = 1
sum(k + 1, n - 1) = sum(6, 6) = 1
 

注意:

1 <= n <= 2000。
给定数组中的元素会在 [-1,000,000, 1,000,000] 范围内。

bool splitArray(int* nums, int numsSize){
    if (numsSize < 7) {
        return false;
    }
    for (int i = 1; i < numsSize; i++) {
        nums[i] = nums[i - 1] + nums[i];
    }

    for (int i = 1; i <= numsSize - 6; i++) {
        int sum1 = nums[i - 1];
        if (sum1 == 0 && nums[numsSize - 1] != 0) {
            continue;
        }
        for (int j = i + 2; j <= numsSize - 4; j++) {
            int sum2 = nums[j - 1] - nums[i];
            if (sum1 != sum2) {
                continue;
            }
            for (int k = j + 2; k <= numsSize - 2; k++) {
                int sum3 = nums[k - 1] - nums[j];
                if (sum2 != sum3) {
                    continue;
                }
                int sum4 = nums[numsSize - 1] - nums[k];
                if (sum3 == sum4) {
                    return true;
                }
            }
        }
    }
    return false;
}

------------------------------------------------------------------------------------

549. 二叉树中最长的连续序列
给定一个二叉树，你需要找出二叉树中最长的连续序列路径的长度。

请注意，该路径可以是递增的或者是递减。例如，[1,2,3,4] 和 [4,3,2,1] 都被认为是合法的，而路径 [1,2,4,3] 则不合法。另一方面，路径可以是 子-父-子 顺序，并不一定是 父-子 顺序。

示例 1:

输入:
        1
       / \
      2   3
输出: 2
解释: 最长的连续路径是 [1, 2] 或者 [2, 1]。
 
示例 2:

输入:
        2
       / \
      1   3
输出: 3
解释: 最长的连续路径是 [1, 2, 3] 或者 [3, 2, 1]。
 
注意: 树上所有节点的值都在 [-1e7, 1e7] 范围内。

int helper(struct TreeNode* node, int diff) {
    if (!node) {
        return 0;
    }
    int left = 0, right = 0;
    if (node->left && node->val - node->left->val == diff) {
        left = 1 + helper(node->left, diff);
    }
    if (node->right && node->val - node->right->val == diff) {
        right = 1 + helper(node->right, diff);
    }
    return left > right ? left : right;
}

int longestConsecutive(struct TreeNode* root) {
    if (!root) {
        return 0;
    }
    int res = helper(root, 1) + helper(root, -1) + 1;
    int leftLen = longestConsecutive(root->left);
    int rightLen = longestConsecutive(root->right);
    int tempLen = leftLen > rightLen ? leftLen : rightLen;
    return res > tempLen ? res : tempLen;
}

------------------------------------------------------------------------------------

555. 分割连接字符串

给定一个字符串列表，你可以将这些字符串连接成一个循环字符串，对于每个字符串，你可以选择是否翻转它。在所有可能的循环字符串中，你需要分割循环字符串（这将使循环字符串变成一个常规的字符串），然后找到字典序最大的字符串。

具体来说，要找到字典序最大的字符串，你需要经历两个阶段：

将所有字符串连接成一个循环字符串，你可以选择是否翻转某些字符串，并按照给定的顺序连接它们。
在循环字符串的某个位置分割它，这将使循环字符串从分割点变成一个常规的字符串。
你的工作是在所有可能的常规字符串中找到字典序最大的一个。

示例:

输入: "abc", "xyz"
输出: "zyxcba"
解释: 你可以得到循环字符串 "-abcxyz-", "-abczyx-", "-cbaxyz-", "-cbazyx-"，
其中 '-' 代表循环状态。 
答案字符串来自第四个循环字符串， 
你可以从中间字符 'a' 分割开然后得到 "zyxcba"。
 

注意:

输入字符串只包含小写字母。
所有字符串的总长度不会超过 1,000。

todo

------------------------------------------------------------------------------------

562. 矩阵中最长的连续1线段
给定一个01矩阵 M，找到矩阵中最长的连续1线段。这条线段可以是水平的、垂直的、对角线的或者反对角线的。

示例:

输入:
[[0,1,1,0],
 [0,1,1,0],
 [0,0,0,1]]
输出: 3
提示: 给定矩阵中的元素数量不会超过 10,000。

typedef struct dir {
    int x;
    int y;
} Dir;

int helpLine(int** M, int MSize, int* MColSize, int * used, Dir *dirc, int k, int i, int j) {
    
    if (i < 0 || i >= MSize || j < 0 || j >= MColSize[0] || M[i][j] == 0) {
        return 0;
    }
    used[(i * MColSize[0] + j) * 4 + k] = 1;
    return helpLine(M, MSize, MColSize, used, dirc, k, i + dirc[k].x, j + dirc[k].y) + 1;
}

int longestLine(int** M, int MSize, int* MColSize) {
    if (MSize == 0 || MColSize[0] == 0) {
        return 0;
    }
    int *used = (int*)malloc(sizeof(int) * MSize * MColSize[0] * 4);
    memset(used, 0, sizeof(int) * MSize * MColSize[0] * 4);
    Dir dirc[4] = { {0, 1}, {1, 0}, {1, 1}, {-1, 1} };  // 右 下 右下 右上
    int res = 0;
    for (int i = 0; i < MSize; i++) {
        for (int j = 0; j < MColSize[0]; j++) {
            if (M[i][j] == 1) {
                for (int k = 0; k < 4; k++) {
                    if (used[(i * MColSize[0] + j) * 4 + k] == 0) {
                        int tempLen = helpLine(M, MSize, MColSize, used, dirc, k, i, j);
                        res = res < tempLen ? tempLen : res;
                        used[(i * MColSize[0] + j) * 4 + k] = 1;
                    }
                }
            }
        }
    }
    free(used);
    return res;
}

------------------------------------------------------------------------------------

573. 松鼠模拟
现在有一棵树，一只松鼠和一些坚果。位置由二维网格的单元格表示。你的目标是找到松鼠收集所有坚果的最小路程，且坚果是一颗接一颗地被放在树下。松鼠一次最多只能携带一颗坚果，松鼠可以向上，向下，向左和向右四个方向移动到相邻的单元格。移动次数表示路程。

输入 1:

输入: 
高度 : 5
宽度 : 7
树的位置 : [2,2]
松鼠 : [4,4]
坚果 : [[3,0], [2,5]]
输出: 12
解释:
​​​​​
注意:

所有给定的位置不会重叠。
松鼠一次最多只能携带一颗坚果。
给定的坚果位置没有顺序。
高度和宽度是正整数。 3 <= 高度 * 宽度 <= 10,000。
给定的网格至少包含一颗坚果，唯一的一棵树和一只松鼠。

int minDistance(int height, int width, int* tree, int treeSize, int* squirrel, int squirrelSize, int** nuts, int nutsSize, int* nutsColSize){
    int res = 0, saveMax = INT_MIN;
    for (int i = 0; i < nutsSize; i++) {
        int lenx = tree[0] > nuts[i][0] ? tree[0] - nuts[i][0] : nuts[i][0] - tree[0];
        int leny = tree[1] > nuts[i][1] ? tree[1] - nuts[i][1] : nuts[i][1] - tree[1];
        int len = lenx + leny;
        res += len * 2;
        lenx = squirrel[0] > nuts[i][0] ? squirrel[0] - nuts[i][0] : nuts[i][0] - squirrel[0];
        leny = squirrel[1] > nuts[i][1] ? squirrel[1] - nuts[i][1] : nuts[i][1] - squirrel[1];
        saveMax = saveMax < len - lenx - leny ? len - lenx - leny : saveMax;
    }
    return res - saveMax;
}

------------------------------------------------------------------------------------

582. 杀死进程
给 n 个进程，每个进程都有一个独一无二的 PID （进程编号）和它的 PPID （父进程编号）。

每一个进程只有一个父进程，但是每个进程可能会有一个或者多个孩子进程。它们形成的关系就像一个树状结构。只有一个进程的 PPID 是 0 ，意味着这个进程没有父进程。所有的 PID 都会是唯一的正整数。

我们用两个序列来表示这些进程，第一个序列包含所有进程的 PID ，第二个序列包含所有进程对应的 PPID。

现在给定这两个序列和一个 PID 表示你要杀死的进程，函数返回一个 PID 序列，表示因为杀这个进程而导致的所有被杀掉的进程的编号。

当一个进程被杀掉的时候，它所有的孩子进程和后代进程都要被杀掉。

你可以以任意顺序排列返回的 PID 序列。

示例 1:

输入: 
pid =  [1, 3, 10, 5]
ppid = [3, 0, 5, 3]
kill = 5
输出: [5,10]
解释: 
           3
         /   \
        1     5
             /
            10
杀掉进程 5 ，同时它的后代进程 10 也被杀掉。
 

注意:

被杀掉的进程编号一定在 PID 序列中。
n >= 1.

todo

------------------------------------------------------------------------------------

616. 给字符串添加加粗标签
给一个字符串 s 和一个字符串列表 dict ，你需要将在字符串列表中出现过的 s 的子串添加加粗闭合标签 <b> 和 </b> 。如果两个子串有重叠部分，你需要把它们一起用一个闭合标签包围起来。同理，如果两个子字符串连续被加粗，那么你也需要把它们合起来用一个加粗标签包围。

样例 1：

输入：
s = "abcxyz123"
dict = ["abc","123"]
输出：
"<b>abc</b>xyz<b>123</b>"
 
样例 2：

输入：
s = "aaabbcc"
dict = ["aaa","aab","bc"]
输出：
"<b>aaabbc</b>c"

注意：
给定的 dict 中不会有重复的字符串，且字符串数目不会超过 100 。
输入中的所有字符串长度都在范围 [1, 1000] 内。

string addBoldTag(string s, vector<string>& dict) {
    string res = "";
    int n = s.size();
    vector<int> flag(n, 0);
    for (string word : dict) {
        int len = word.size();
        for (int i = 0; i <= n - len; ++i) {
            if (s[i] == word[0] && s.substr(i, len) == word) {
                for (int j = i; j < i + len; ++j) {
                    flag[j] = 1;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (flag[i] == 1 && (i == 0 || flag[i - 1] == 0) ) {
            res += "<b>";
        }
        res += s[i];
        if (flag[i] == 1 && (i == n - 1 || flag[i + 1] == 0)) {
            res += "</b>";
        }
    }
    return res;
}

------------------------------------------------------------------------------------

625. 最小因式分解
给定一个正整数 a，找出最小的正整数 b 使得 b 的所有数位相乘恰好等于 a。

如果不存在这样的结果或者结果不是 32 位有符号整数，返回 0。

样例 1

输入：

48 
输出：

68

样例 2

输入：

15
输出：

35

long long strToLonglong(string s) {
    long long res = 0;
    for (char c : s) {
        res = c - '0' + res * 10;
    }
    return res;
}

int smallestFactorization(int a) {
    if (a == 1) {
        return 1;
    }
    string res = "";
    for (int k = 9; k >= 2; --k) {
        while (a % k == 0) {
            res = (char)('0' + k) + res;
            a /= k;
        }
    }
    if (a > 1) {
        return 0;
    }
    long long num = strToLonglong(res);
    return num > INT_MAX ? 0 : num;
}

------------------------------------------------------------------------------------

634. 寻找数组的错位排列
在组合数学中，如果一个排列中所有元素都不在原先的位置上，那么这个排列就被称为错位排列。

给定一个从 1 到 n 升序排列的数组，你可以计算出总共有多少个不同的错位排列吗？

由于答案可能非常大，你只需要将答案对 109+7 取余输出即可。

样例 1:

输入: 3
输出: 2
解释: 原始的数组为 [1,2,3]。两个错位排列的数组为 [2,3,1] 和 [3,1,2]。
 
int findDerangement(int n){
    if (n <= 1) {
        return 0;
    }
    else if (n == 2) {
        return 1;
    }
    long long pre = 1, prepre = 0, res = 0;
    for (int i = 3; i <= n; i++) {
        res = (i - 1) * (pre + prepre) % 1000000007;
        prepre = pre;
        pre = res;
    }
    return res;
}

------------------------------------------------------------------------------------

635. 设计日志存储系统
你将获得多条日志，每条日志都有唯一的 id 和 timestamp，timestamp 是形如 Year:Month:Day:Hour:Minute:Second 的字符串，例如 2017:01:01:23:59:59，所有值域都是零填充的十进制数。

设计一个日志存储系统实现如下功能：

void Put(int id, string timestamp)：给定日志的 id 和 timestamp，将这个日志存入你的存储系统中。

int[] Retrieve(String start, String end, String granularity)：返回在给定时间区间内的所有日志的 id。start 、 end 和 timestamp 的格式相同，granularity 表示考虑的时间级。比如，start = "2017:01:01:23:59:59", end = "2017:01:02:23:59:59", granularity = "Day" 代表区间 2017 年 1 月 1 日到 2017 年 1 月 2 日。

样例 1 ：

put(1, "2017:01:01:23:59:59");
put(2, "2017:01:01:22:59:59");
put(3, "2016:01:01:00:00:00");
retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Year"); // 返回值 [1,2,3]，返回从 2016 年到 2017 年所有的日志。
retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Hour"); // 返回值 [1,2], 返回从 2016:01:01:01 到 2017:01:01:23 区间内的日志，日志 3 不在区间内。
 
注释 ：

Put 和 Retrieve 的指令总数不超过 300。
年份的区间是 [2000,2017]，小时的区间是 [00,23]。
Retrieve 的输出顺序不作要求。

class LogSystem {
public:
    LogSystem() {
        units = {"Year", "Month", "Day", "Hour", "Minute", "Second"};
        indices = {4, 7, 10, 13, 16, 19}; 
    }
    
    void put(int id, string timestamp) {
        timestamps.push_back({id, timestamp});
    }
    
    vector<int> retrieve(string s, string e, string gra) {
        vector<int> res;
        int idx = indices[find(units.begin(), units.end(), gra) - units.begin()];
        for (auto p : timestamps) {
            string t = p.second;
            if (t.substr(0, idx).compare(s.substr(0, idx)) >= 0 && t.substr(0, idx).compare(e.substr(0, idx)) <= 0) {
                res.push_back(p.first);
            }
        }
        return res;
    }

private:
    vector<pair<int, string>> timestamps;
    vector<string> units;
    vector<int> indices;
};

------------------------------------------------------------------------------------

651. 4键键盘
假设你有一个特殊的键盘包含下面的按键：

Key 1: (A)：在屏幕上打印一个 'A'。

Key 2: (Ctrl-A)：选中整个屏幕。

Key 3: (Ctrl-C)：复制选中区域到缓冲区。

Key 4: (Ctrl-V)：将缓冲区内容输出到上次输入的结束位置，并显示在屏幕上。

现在，你只可以按键 N 次（使用上述四种按键），请问屏幕上最多可以显示几个 'A'呢？

样例 1:

输入: N = 3
输出: 3
解释: 
我们最多可以在屏幕上显示三个'A'通过如下顺序按键：
A, A, A
 

样例 2:

输入: N = 7
输出: 9
解释: 
我们最多可以在屏幕上显示九个'A'通过如下顺序按键：
A, A, A, Ctrl A, Ctrl C, Ctrl V, Ctrl V
 

注释:

1 <= N <= 50
结果不会超过 32 位有符号整数范围。
 
int maxA(int N){
    int *dp = (int*)malloc(sizeof(int) * (N + 1));
    for (int i = 1; i <= N; ++i) {  // 打i次
        dp[i] = i;
        for (int j = 1; j <= i - 3; ++j) {  // 假设已经算过的个数是j，剩下全粘贴
            int temp = dp[j] * (i - 2 - j + 1);
            dp[i] = dp[i] > temp ? dp[i] : temp;
        }
    }
    return dp[N];
}

------------------------------------------------------------------------------------

663. 均匀树划分
给定一棵有 n 个结点的二叉树，你的任务是检查是否可以通过去掉树上的一条边将树分成两棵，且这两棵树结点之和相等。

样例 1:

输入:     
    5
   / \
  10 10
    /  \
   2   3

输出: True
解释: 
    5
   / 
  10
      
和: 15

   10
  /  \
 2    3

和: 15
 

样例 2:

输入:     
    1
   / \
  2  10
    /  \
   2   20

输出: False
解释: 无法通过移除一条树边将这棵树划分成结点之和相等的两棵子树。
 

注释 :

树上结点的权值范围 [-100000, 100000]。
1 <= n <= 10000

int helperTree(TreeNode* node, map<int, int>& m) {
    if (!node) {
        return 0;
    }
    int cur = node->val + helperTree(node->left, m) + helperTree(node->right, m);
    ++m[cur];
    return cur;
}

bool checkEqualTree(TreeNode* root) {
    map<int, int> m;
    int sum = helperTree(root, m);
    if (sum == 0) {
        return m[0] > 1;
    }
    return (sum % 2 == 0) && m.count(sum / 2);
}

------------------------------------------------------------------------------------

666. 路径和 IV
对于一棵深度小于 5 的树，可以用一组三位十进制整数来表示。

对于每个整数：

百位上的数字表示这个节点的深度 D，1 <= D <= 4。
十位上的数字表示这个节点在当前层所在的位置 P， 1 <= P <= 8。位置编号与一棵满二叉树的位置编号相同。
个位上的数字表示这个节点的权值 V，0 <= V <= 9。
给定一个包含三位整数的升序数组，表示一棵深度小于 5 的二叉树，请你返回从根到所有叶子结点的路径之和。

样例 1:

输入: [113, 215, 221]
输出: 12
解释: 
这棵树形状如下:
    3
   / \
  5   1

路径和 = (3 + 5) + (3 + 1) = 12.

样例 2:

输入: [113, 221]
输出: 4
解释: 
这棵树形状如下: 
    3
     \
      1

路径和 = (3 + 1) = 4.

todo
------------------------------------------------------------------------------------

681. 最近时刻
给定一个形如 “HH:MM” 表示的时刻，利用当前出现过的数字构造下一个距离当前时间最近的时刻。每个出现数字都可以被无限次使用。

你可以认为给定的字符串一定是合法的。例如，“01:34” 和 “12:09” 是合法的，“1:34” 和 “12:9” 是不合法的。

样例 1:

输入: "19:34"
输出: "19:39"
解释: 利用数字 1, 9, 3, 4 构造出来的最近时刻是 19:39，是 5 分钟之后。结果不是 19:33 因为这个时刻是 23 小时 59 分钟之后。
 
样例 2:

输入: "23:59"
输出: "22:22"
解释: 利用数字 2, 3, 5, 9 构造出来的最近时刻是 22:22。 答案一定是第二天的某一时刻，所以选择可构造的最小时刻。

int compareTime(const void *a, const void *b) {
    return *((char*)a) > *((char*)b);
}

int findInOrder(char *timeOrdered, char c) {
    int flag = 0;
    for (int i = 0; i < 4; i++) {
        if (flag == 1 && timeOrdered[i] != timeOrdered[i - 1]) {
            return i - 1;
        }
        if (timeOrdered[i] == c) {
            flag = 1;
        }        
    }
    return -1;
}

char * nextClosestTime(char * time){
    int size = strlen(time);
    char *res = (char*)malloc(sizeof(char) * (size + 1));
    strncpy(res, time, size);
    res[size] = 0;

    char timeOrdered[4] = { time[0], time[1], time[3], time[4] };
    qsort(timeOrdered, 4, sizeof(char), compareTime);
    if (res[4] != timeOrdered[3]) {
        res[4] = timeOrdered[findInOrder(timeOrdered, res[4]) + 1];        
        return res;
    }
    res[4] = timeOrdered[0];
    
    int index = findInOrder(timeOrdered, res[3]);
    if (res[3] != timeOrdered[3] && timeOrdered[index + 1] < '6') {
        res[3] = timeOrdered[index + 1];
        return res;
    }
    else {
        res[3] = timeOrdered[0];
    }
    
    index = findInOrder(timeOrdered, res[1]);
    if (res[1] != timeOrdered[3] && (res[0] < '2' || (res[0] == '2' && timeOrdered[index + 1] <= '4'))) {
        res[1] = timeOrdered[index + 1];
        return res;
    }
    else {
        res[1] = timeOrdered[0];
    }

    index = findInOrder(timeOrdered, res[0]);
    if (res[0] != timeOrdered[3] && timeOrdered[index + 1] <= '2') {
        res[0] = timeOrdered[index + 1];
        return res;
    }
    else {
        res[0] = timeOrdered[0];
    }
    return res;
}

------------------------------------------------------------------------------------

694. 不同岛屿的数量
给定一个非空01二维数组表示的网格，一个岛屿由四连通（上、下、左、右四个方向）的 1 组成，你可以认为网格的四周被海水包围。

请你计算这个网格中共有多少个形状不同的岛屿。两个岛屿被认为是相同的，当且仅当一个岛屿可以通过平移变换（不可以旋转、翻转）和另一个岛屿重合。

样例 1:

11000
11000
00011
00011
给定上图，返回结果 1。

样例 2:

11011
10000
00001
11011
给定上图，返回结果 <font color="#c7254e" face="Menlo, Monaco, Consolas, Courier New, monospace">3</font>。

注意:

11
1
和

 1
11
是不同的岛屿，因为我们不考虑旋转、翻转操作。

注释 :  二维数组每维的大小都不会超过50。

public:
    void dfs(vector<int> & island, vector<vector<bool>> &visited, vector<vector<int>>& grid, int sr, int sc, int r, int c) {
    if (r >= grid.size() || r < 0 || c >= grid[0].size() || c < 0 || visited[r][c] || !grid[r][c]) return;
    visited[r][c] = true;
    island.push_back((r - sr)*grid[0].size() + (c - sc));
    dfs(island, visited, grid, sr, sc, r - 1, c);
    dfs(island, visited, grid, sr, sc, r + 1, c);
    dfs(island, visited, grid, sr, sc, r, c - 1);
    dfs(island, visited, grid, sr, sc, r, c + 1);
}

int numDistinctIslands(vector<vector<int>>& grid) {
    vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
    set<vector<int>> islands;
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[0].size(); j++) {
            vector<int> island;
            dfs(island, visited, grid, i, j, i, j);
            if (!island.empty()) islands.insert(island);
        }
    }
    return islands.size();
}

------------------------------------------------------------------------------------

702. 搜索长度未知的有序数组
给定一个升序整数数组，写一个函数搜索 nums 中数字 target。如果 target 存在，返回它的下标，否则返回 -1。注意，这个数组的大小是未知的。你只可以通过 ArrayReader 接口访问这个数组，ArrayReader.get(k) 返回数组中第 k 个元素（下标从 0 开始）。

你可以认为数组中所有的整数都小于 10000。如果你访问数组越界，ArrayReader.get 会返回 2147483647。

样例 1：

输入: array = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 存在在 nums 中，下标为 4
样例 2：

输入: array = [-1,0,3,5,9,12], target = 2
输出: -1
解释: 2 不在数组中所以返回 -1
 
注释 ：

你可以认为数组中所有元素的值互不相同。
数组元素的值域是 [-9999, 9999]。

int search(const ArrayReader& reader, int target) {
    int left = 0, right = INT_MAX, mid, midValue;
    while (left <= right) {
        mid = left + (right - left) / 2;
        midValue = reader.get(mid);
        if (target == midValue) {
            return mid;
        }
        else if (target < midValue) {
            right = mid - 1;
        }
        else {
            left = mid + 1;
        }
    }
    return -1;
}

------------------------------------------------------------------------------------

708. 循环有序列表的插入

给定循环升序列表中的一个点，写一个函数向这个列表中插入一个新元素，使这个列表仍然是循环升序的。给定的可以是这个列表中任意一个顶点的指针，并不一定是这个列表中最小元素的指针。

如果有多个满足条件的插入位置，你可以选择任意一个位置插入新的值，插入后整个列表仍然保持有序。

如果列表为空（给定的节点是 null），你需要创建一个循环有序列表并返回这个点。否则。请返回原先给定的节点。

下面的例子可以帮你更好的理解这个问题：

 

在上图中，有一个包含三个元素的循环有序列表，你获得值为 3 的节点的指针，我们需要向表中插入元素 2。

新插入的节点应该在 1 和 3 之间，插入之后，整个列表如上图所示，最后返回节点 3。

Node* insert(Node* head, int insertVal) {
    if (head == NULL) {
        Node *n = new Node(insertVal, NULL);
        n->next = n;
        return n;
    }
    Node *h = head;
    while (!(insertVal <= h->next->val && insertVal >= h->val) &&
           !(h->next->val < h->val && (insertVal <= h->next->val || insertVal >= h->val)) &&
            h->next != head) {
        h = h->next;
    }
    Node *n = new Node(insertVal, h->next);
    h->next = n; 
    return head;
}

------------------------------------------------------------------------------------

723. 粉碎糖果
这个问题是实现一个简单的消除算法。

给定一个二维整数数组 board 代表糖果所在的方格，不同的正整数 board[i][j] 代表不同种类的糖果，如果 board[i][j] = 0 代表 (i, j) 这个位置是空的。给定的方格是玩家移动后的游戏状态，现在需要你根据以下规则粉碎糖果，使得整个方格处于稳定状态并最终输出。

如果有三个及以上水平或者垂直相连的同种糖果，同一时间将它们粉碎，即将这些位置变成空的。
在同时粉碎掉这些糖果之后，如果有一个空的位置上方还有糖果，那么上方的糖果就会下落直到碰到下方的糖果或者底部，这些糖果都是同时下落，也不会有新的糖果从顶部出现并落下来。
通过前两步的操作，可能又会出现可以粉碎的糖果，请继续重复前面的操作。
当不存在可以粉碎的糖果，也就是状态稳定之后，请输出最终的状态。
你需要模拟上述规则并使整个方格达到稳定状态，并输出。

样例 :

输入:
board = 
[[110,5,112,113,114],[210,211,5,213,214],[310,311,3,313,314],[410,411,412,5,414],[5,1,512,3,3],[610,4,1,613,614],[710,1,2,713,714],[810,1,2,1,1],[1,1,2,2,2],[4,1,4,4,1014]]

输出:
[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[110,0,0,0,114],[210,0,0,0,214],[310,0,0,113,314],[410,0,0,213,414],[610,211,112,313,614],[710,311,412,613,714],[810,411,512,713,1014]]

解释: 

 

注释 :

board 数组的行数区间是 [3, 50]。
board[i] 数组的长度区间（即 board 数组的列数区间）是 [3, 50]。
每个 board[i][j] 初始整数范围是 [1, 2000]。

vector<vector<int>> candyCrush(vector<vector<int>>& board) {
    int m = board.size(), n = board[0].size();
    while (true) {
        vector<int> flag(m * n, 1);
        int flagCnt = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 0) {
                    continue;
                }
                int x0 = i, x1 = i;
                while (x0 >= 0 && board[x0][j] == board[i][j]) {
                    x0--;
                }
                while (x1 < m && board[x1][j] == board[i][j]) {
                    x1++;
                }
                if (x1 - x0 >= 4) {
                    flagCnt = 1;
                    for (int k = i; k > x0; k--) {
                        flag[k * n + j] = 0;
                    }
                    for (int k = i + 1; k < x1; k++) {
                        flag[k * n + j] = 0;
                    }
                }

                int y0 = j, y1 = j;
                while (y0 >= 0 && board[i][y0] == board[i][j]) {
                    y0--;
                }
                while (y1 < n && board[i][y1] == board[i][j]) {
                    y1++;
                }
                if (y1 - y0 >= 4) {
                    flagCnt = 1;                    
                    for (int k = j; k > y0; k--) {
                        flag[i * n + k] = 0;
                    }
                    for (int k = j + 1; k < y1; k++) {
                        flag[i * n + k] = 0;
                    }
                }                
            }
        }
        if (flagCnt == 0) {
            break;
        }        
        for (int j = 0; j < n; j++) {
            int temp = m - 1;
            for (int i = m - 1; i >= 0; i--) {
                if (flag[i * n + j] == 1) {
                    board[temp][j] = board[i][j];
                    if(temp != i) {
                        board[i][j] = 0;
                    }
                    temp--;
                }
                else {
                    board[i][j] = 0;
                }
            }
        }
    }
    return board;
}

------------------------------------------------------------------------------------

737. 句子相似性 II
给定两个句子 words1, words2 （每个用字符串数组表示），和一个相似单词对的列表 pairs ，判断是否两个句子是相似的。

例如，当相似单词对是 pairs = [["great", "fine"], ["acting","drama"], ["skills","talent"]]的时候，words1 = ["great", "acting", "skills"] 和 words2 = ["fine", "drama", "talent"] 是相似的。

注意相似关系是 具有 传递性的。例如，如果 "great" 和 "fine" 是相似的，"fine" 和 "good" 是相似的，则 "great" 和 "good" 是相似的。

而且，相似关系是具有对称性的。例如，"great" 和 "fine" 是相似的相当于 "fine" 和 "great" 是相似的。

并且，一个单词总是与其自身相似。例如，句子 words1 = ["great"], words2 = ["great"], pairs = [] 是相似的，尽管没有输入特定的相似单词对。

最后，句子只会在具有相同单词个数的前提下才会相似。所以一个句子 words1 = ["great"] 永远不可能和句子 words2 = ["doubleplus","good"] 相似。

注：

words1 and words2 的长度不会超过 1000。
pairs 的长度不会超过 2000。
每个pairs[i] 的长度为 2。
每个 words[i] 和 pairs[i][j] 的长度范围为 [1, 20]。

bool areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs) {
    if (words1.size() != words2.size()) {
        return false;
    }
    unordered_map<string, unordered_set<string>> m;
    for (auto pair : pairs) {
        m[pair[0]].insert(pair[1]);
        m[pair[1]].insert(pair[0]);
    }
    for (int i = 0; i < words1.size(); ++i) {
        if (words1[i] == words2[i]) {
            continue;
        }
        unordered_set<string> visited;
        queue<string> q{ {words1[i]} };
        bool succ = false;
        while (!q.empty()) {
            auto t = q.front();
            q.pop();
            if (m[t].count(words2[i])) {
                succ = true;
                break;
            }
            visited.insert(t);
            for (auto a : m[t]) {
                if (!visited.count(a)) {
                    q.push(a);
                }
            }
        }
        if (!succ) {
            return false;
        }
    }
    return true;
}

------------------------------------------------------------------------------------

742. 二叉树最近的叶节点
给定一个 每个结点的值互不相同 的二叉树，和一个目标值 k，找出树中与目标值 k 最近的叶结点。 

这里，与叶结点 最近 表示在二叉树中到达该叶节点需要行进的边数与到达其它叶结点相比最少。而且，当一个结点没有孩子结点时称其为叶结点。

在下面的例子中，输入的树以逐行的平铺形式表示。实际上的有根树 root 将以TreeNode对象的形式给出。

示例 1：

输入：
root = [1, 3, 2], k = 1
二叉树图示：
          1
         / \
        3   2

输出： 2 (或 3)

解释： 2 和 3 都是距离目标 1 最近的叶节点。
 
示例 2：

输入：
root = [1], k = 1
输出：1

解释： 最近的叶节点是根结点自身。
 
示例 3：

输入：
root = [1,2,3,4,null,null,null,5,null,6], k = 2
二叉树图示：
             1
            / \
           2   3
          /
         4
        /
       5
      /
     6

输出：3
解释： 值为 3（而不是值为 6）的叶节点是距离结点 2 的最近结点。
 
注：

root 表示的二叉树最少有 1 个结点且最多有 1000 个结点。
每个结点都有一个唯一的 node.val ，范围为 [1, 1000]。
给定的二叉树中有某个结点使得 node.val == k。
 
int findClosestLeaf(TreeNode* root, int k) {
    unordered_map<TreeNode*, TreeNode*> back;
    TreeNode *kNode = find(root, k, back);
    queue<TreeNode*> q{{kNode}};
    unordered_set<TreeNode*> visited{{kNode}};
    while (!q.empty()) {
        TreeNode *t = q.front(); q.pop();
        if (!t->left && !t->right) return t->val;
        if (t->left && !visited.count(t->left)) {
            visited.insert(t->left);
            q.push(t->left);
        }
        if (t->right && !visited.count(t->right)) {
            visited.insert(t->right);
            q.push(t->right);
        }
        if (back.count(t) && !visited.count(back[t])) {
            visited.insert(back[t]);
            q.push(back[t]);
        }
    }
    return -1;
}
TreeNode* find(TreeNode* node, int k, unordered_map<TreeNode*, TreeNode*>& back) {
    if (node->val == k) return node;
    if (node->left) {
        back[node->left] = node;
        TreeNode *left = find(node->left, k, back);
        if (left) return left;
    }
    if (node->right) {
        back[node->right] = node;
        TreeNode *right = find(node->right, k, back);
        if (right) return right;
    }
    return NULL;
}

------------------------------------------------------------------------------------

750. 角矩形的数量
给定一个只包含 0 和 1 的网格，找出其中角矩形的数量。

一个 角矩形 是由四个不同的在网格上的 1 形成的轴对称的矩形。注意只有角的位置才需要为 1。并且，4 个 1 需要是不同的。

示例 1：

输入：grid = 
[[1, 0, 0, 1, 0],
 [0, 0, 1, 0, 1],
 [0, 0, 0, 1, 0],
 [1, 0, 1, 0, 1]]
输出：1
解释：只有一个角矩形，角的位置为 grid[1][2], grid[1][4], grid[3][2], grid[3][4]。
 
示例 2：

输入：grid = 
[[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]]
输出：9
解释：这里有 4 个 2x2 的矩形，4 个 2x3 和 3x2 的矩形和 1 个 3x3 的矩形。
 
示例 3：

输入：grid = 
[[1, 1, 1, 1]]
输出：0
解释：矩形必须有 4 个不同的角。
 
注：

网格 grid 中行和列的数目范围为 [1, 200]。
Each grid[i][j] will be either 0 or 1.
网格中 1 的个数不会超过 6000。

int countCornerRectangles(int** grid, int gridSize, int* gridColSize){
	int res = 0;
	for (int i = 0; i < gridSize; i++) {
		for (int j = 0; j < gridColSize[0]; j++) {
			if (grid[i][j] == 0) {
				continue;
			}
			for (int k = j + 1; k < gridColSize[0]; k++) {
                if (grid[i][k] == 0) {
                    continue;
                }
                for (int l = i + 1; l < gridSize; l++) {
                    if (grid[l][j] == 1 && grid[l][k] == 1) {
                        res++;
                    }
                }
			}
		}
	}
	return res;
}

------------------------------------------------------------------------------------

755. 倒水
给出一个地形高度图， heights[i] 表示该索引处的高度。每个索引的宽度为 1。在 V 个单位的水落在索引 K 处以后，每个索引位置有多少水？

水最先会在索引 K 处下降并且落在该索引位置的最高地形或水面之上。然后按如下方式流动：

如果液滴最终可以通过向左流动而下降，则向左流动。
否则，如果液滴最终可以通过向右流动而下降，则向右流动。
否则，在当前的位置上升。
这里，“最终下降” 的意思是液滴如果按此方向移动的话，最终可以下降到一个较低的水平。而且，“水平”的意思是当前列的地形的高度加上水的高度。

我们可以假定在数组两侧的边界外有无限高的地形。而且，不能有部分水在多于 1 个的网格块上均匀分布 - 每个单位的水必须要位于一个块中。

示例 1：

输入：heights = [2,1,1,2,1,2,2], V = 4, K = 3
输出：[2,2,2,3,2,2,2]
解释：
#       #
#       #
##  # ###
#########
 0123456    <- 索引

第一个水滴降落在索引 K = 3 上：

#       #
#   w   #
##  # ###
#########
 0123456    

当向左或向右移动时，水可以移动到相同或更低的高度。When moving left or right, the water can only move to the same level or a lower level.
（从水平上看，意思是该列的地形高度加上水的高度）
由于向左移动可以最终下落，因此向左移动。
（一个水滴 “下落” 的意思是可以相比之前可以进入更低的高度）

#       #
#       #
## w# ###
#########
 0123456    

由于向左移动不会使其降落，所以停在该位置上。下一个水滴下落：

#       #
#   w   #
## w# ###
#########
 0123456  


由于新水滴向左移动可以最终下落，因此向左移动。
注意水滴仍然是优先选择向左移动，
尽管可以向右移动（而且向右移动可以下落更快）


#       #
#  w    #
## w# ###
#########
 0123456  

#       #
#       #
##ww# ###
#########
 0123456  

经过刚才的阶段后，第三个水滴下落。
由于向左移动不会最终下落，因此尝试向右移动。
由于向右移动可以最终下落，因此向右移动。


#       #
#   w   #
##ww# ###
#########
 0123456  

#       #
#       #
##ww#w###
#########
 0123456  

最终，第四个水滴下落。
由于向左移动不会最终下落，因此尝试向右移动。
由于向右移动不会最终下落，因此停在当前位置：

#       #
#   w   #
##ww#w###
#########
 0123456  

最终的答案为 [2,2,2,3,2,2,2]:

    #    
 ####### 
 ####### 
 0123456 
 
示例 2：

输入：heights = [1,2,3,4], V = 2, K = 2
输出：[2,3,3,4]
解释：
最后的水滴落在索引 1 位置，因为继续向左移动不会使其下降到更低的高度。
 
示例 3：

输入：heights = [3,1,3], V = 5, K = 1
输出：[4,4,4]
 

注：
heights 的长度为 [1, 100] ，并且每个数的范围为[0, 99]。
V 的范围 [0, 2000]。
K 的范围 [0, heights.length - 1]。

vector<int> pourWater(vector<int>& heights, int V, int K) {
    for (int i = 0; i < V; ++i) {
        int l = K, r = K, n = heights.size();
        while (l > 0 && heights[l] >= heights[l - 1]) --l;
        while (l < K && heights[l] == heights[l + 1]) ++l;
        if (l != K) {
            heights[l]++;
            continue;
        }
        while (r < n - 1 && heights[r] >= heights[r + 1]) ++r;
        while (r > K && heights[r] == heights[r - 1]) --r;
        heights[r]++;
    }
    return heights;
}

------------------------------------------------------------------------------------

776. 拆分二叉搜索树
给你一棵二叉搜索树（BST）、它的根结点 root 以及目标值 V。

请将该树按要求拆分为两个子树：其中一个子树结点的值都必须小于等于给定的目标值 V；另一个子树结点的值都必须大于目标值 V；树中并非一定要存在值为 V 的结点。

除此之外，树中大部分结构都需要保留，也就是说原始树中父节点 P 的任意子节点 C，假如拆分后它们仍在同一个子树中，那么结点 P 应仍为 C 的子结点。

你需要返回拆分后两个子树的根结点 TreeNode，顺序随意。

示例 1：

输入：root = [4,2,6,1,3,5,7], V = 2
输出：[[2,1],[4,3,6,null,null,5,7]]
解释：
注意根结点 output[0] 和 output[1] 都是 TreeNode 对象，不是数组。

给定的树 [4,2,6,1,3,5,7] 可化为如下示意图：

          4
        /   \
      2      6
     / \    / \
    1   3  5   7

输出的示意图如下：

          4
        /   \
      3      6       和    2
            / \           /
           5   7         1
注意：

二叉搜索树节点个数不得超过 50 个
二叉搜索树始终是有效的，并且每个节点的值dku bu xiang t

给你一棵二叉搜索树（BST）、它的根结点 root 以及目标值 V。

请将该树按要求拆分为两个子树：其中一个子树结点的值都必须小于等于给定的目标值 V；另一个子树结点的值都必须大于目标值 V；树中并非一定要存在值为 V 的结点。

除此之外，树中大部分结构都需要保留，也就是说原始树中父节点 P 的任意子节点 C，假如拆分后它们仍在同一个子树中，那么结点 P 应仍为 C 的子结点。

你需要返回拆分后两个子树的根结点 TreeNode，顺序随意。

示例 1：

输入：root = [4,2,6,1,3,5,7], V = 2
输出：[[2,1],[4,3,6,null,null,5,7]]
解释：
注意根结点 output[0] 和 output[1] 都是 TreeNode 对象，不是数组。

给定的树 [4,2,6,1,3,5,7] 可化为如下示意图：

          4
        /   \
      2      6
     / \    / \
    1   3  5   7

输出的示意图如下：

          4
        /   \
      3      6       和    2
            / \           /
           5   7         1
注意：

二叉搜索树节点个数不得超过 50 个
二叉搜索树始终是有效的，并且每个节点的值dku bu xiang t

vector<TreeNode*> splitBST(TreeNode* root, int V) {
    vector<TreeNode*> res{NULL, NULL};
    if (!root) return res;
    if (root->val <= V) {
        res = splitBST(root->right, V);
        root->right = res[0];
        res[0] = root;
    } else {
        res = splitBST(root->left, V);
        root->left = res[1];
        res[1] = root;
    }
    return res;
}

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------