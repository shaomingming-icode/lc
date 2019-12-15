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
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------


