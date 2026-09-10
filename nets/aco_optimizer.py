"""基于蚁群路径搜索的分割概率图后处理。"""


import numpy as np
from tqdm import trange


class AntColonyOptimizer:
    """以预测概率为启发信息，通过蚁群路径搜索更新裂缝信息素图。"""

    def __init__(self,
                 num_ants: int = 2000,
                 max_iter: int = 50,
                 alpha: float = 1.0,
                 beta: float = 3.0,
                 rho: float = 0.1,
                 q: float = 1.0,
                 min_path_len: int = 15,
                 start_threshold: float = 0.8):
        """配置蚂蚁数量、路径搜索与信息素更新参数。

        Args:
            num_ants: 每轮投放的蚂蚁数量。
            max_iter: 信息素更新的总迭代次数。
            alpha: 信息素在转移概率中的指数。
            beta: 原始预测概率在转移概率中的指数。
            rho: 每轮信息素蒸发率。
            q: 每条保留路径增加的信息素总量。
            min_path_len: 最大移动步数，同时作为保留路径的最小节点数。
            start_threshold: 候选起点必须超过的预测概率阈值。
        """
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.min_path_len = min_path_len
        self.start_threshold = start_threshold

    def run(self, probability_map: np.ndarray) -> np.ndarray:
        """对二维目标概率图执行蚁群优化，返回更新后的信息素图。

        输入概率范围为 [0, 1]；返回值是累计信息素强度，不保证仍在 [0, 1]。
        """
        height, width = probability_map.shape

        # 以概率图作为启发信息，并加入微小常数避免零值。
        self.heuristic_info = probability_map + 1e-10

        # 使用概率图副本初始化信息素图。
        self.pheromone_map = probability_map.copy()

        # 根据阈值寻找所有可能的蚂蚁起始点
        start_points = np.argwhere(probability_map > self.start_threshold)
        if len(start_points) == 0:
            print(
                f"Warning: No start points found above threshold {self.start_threshold}. Using the single highest probability point.")
            # 无像素超过阈值时，使用全图概率最高的像素作为起点。
            start_points = np.array([np.unravel_index(np.argmax(probability_map), probability_map.shape)])

        print(f"Found {len(start_points)} possible start locations for ants.")

        # 主循环
        for it in trange(self.max_iter, desc="ACO Iterations"):
            all_paths = []

            # 从候选起点中有放回采样 num_ants 个位置。
            ant_start_indices = np.random.choice(len(start_points), self.num_ants)
            ant_starts = start_points[ant_start_indices]

            # 为每只蚂蚁构建路径，并筛选满足长度要求的结果。
            for y_start, x_start in ant_starts:
                path = self._construct_path((y_start, x_start), height, width)
                if len(path) >= self.min_path_len:
                    all_paths.append(path)

            # 根据本轮保留路径更新信息素。
            self._update_pheromone(all_paths)

        return self.pheromone_map

    def _construct_path(self, start_pos: tuple, H: int, W: int) -> list:
        """从起点执行最多 min_path_len 次邻域移动，遇到无可用邻居时提前结束。"""
        path = [start_pos]
        tabu_list = {start_pos}  # 记录已访问坐标，避免路径重复访问同一像素。

        current_pos = start_pos

        # 最多移动 min_path_len 步；无法继续时提前结束。
        for _ in range(self.min_path_len):
            neighbors, valid_mask = self._get_neighbors(current_pos, H, W, tabu_list)

            # 如果没有可走的邻居，路径构建结束
            if not np.any(valid_mask):
                break

            # 计算并获取前往各个有效邻居的转移概率
            transition_probs = self._calculate_transition_probabilities(neighbors, valid_mask)

            # 使用轮盘赌选择法，根据概率决定下一个位置
            valid_indices = np.where(valid_mask)[0]
            next_pos_local_idx = np.random.choice(len(valid_indices), p=transition_probs)
            next_pos_global_idx = valid_indices[next_pos_local_idx]
            next_pos = tuple(neighbors[next_pos_global_idx])

            path.append(next_pos)
            tabu_list.add(next_pos)
            current_pos = next_pos

        return path

    def _get_neighbors(self, pos: tuple, H: int, W: int, tabu_list: set) -> tuple:
        """返回八邻域中未越界、未访问的坐标及其有效性掩码。"""
        y, x = pos
        # 定义8个方向的偏移量
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbors = []
        valid_mask = []

        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            is_valid = (0 <= ny < H) and (0 <= nx < W) and ((ny, nx) not in tabu_list)
            if is_valid:
                neighbors.append((ny, nx))
                valid_mask.append(True)

        return np.array(neighbors), np.array(valid_mask)

    def _calculate_transition_probabilities(self, neighbors: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """根据邻居的信息素和启发信息计算归一化转移概率。"""
        valid_neighbors = neighbors
        if len(valid_neighbors) == 0:
            return np.array([])

        ny, nx = valid_neighbors[:, 0], valid_neighbors[:, 1]

        # P_ij = (tau_j^alpha * eta_j^beta) / sum(tau_k^alpha * eta_k^beta)
        pheromone_values = self.pheromone_map[ny, nx] ** self.alpha
        heuristic_values = self.heuristic_info[ny, nx] ** self.beta

        scores = pheromone_values * heuristic_values

        total_score = np.sum(scores)
        if total_score == 0:
            # 如果所有邻居分数都为0，则等概率选择一个
            return np.ones(len(scores)) / len(scores)

        return scores / total_score

    def _update_pheromone(self, all_paths: list):
        """蒸发现有信息素，并为保留路径上的像素增加信息素。"""
        # 按蒸发率衰减已有信息素。
        self.pheromone_map *= (1 - self.rho)

        # 仅对保留路径增加信息素。
        if not all_paths:
            return

        for path in all_paths:
            # 将每条路径的固定信息素总量均分到路径像素。
            pheromone_deposit = self.q / len(path)

            # 将路径上的点坐标提取出来
            path_coords = np.array(path)
            rows, cols = path_coords[:, 0], path_coords[:, 1]

            # 通过 NumPy 高级索引更新整条路径的信息素。
            self.pheromone_map[rows, cols] += pheromone_deposit

