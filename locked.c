
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


------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
