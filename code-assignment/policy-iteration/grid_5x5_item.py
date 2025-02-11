import numpy as np

np.random.seed(0)

# -------------------------------------------------------------
# 1. KHỞI TẠO VÀ THIẾT LẬP THAM SỐ BAN ĐẦU
# -------------------------------------------------------------
## Xây dựng môi trường (state) grid 5x5
grid_size = 5
gamma = 0.9
goal_state = (4, 4)
obstacle = [(1, 1), (2, 3), (4, 3)]
speed = [(2,1), (3, 2), (3, 4)]

## Xây dựng hàm giá trị ban đầu
V = np.zeros((grid_size, grid_size))

## Khởi tạo chính sách ban đầu một cách ngẫu nhiên (mỗi ô có 1 hành động trong 4 hướng: 0: xuống, 1: lên, 2: trái, 3: phải)
policy = np.random.choice(4, size = (grid_size, grid_size))
print(policy)

# -------------------------------------------------------------
# 2. XÂY DỰNG HÀM TAKE_ACTION
# -------------------------------------------------------------

# Xây dựng hàm Take_Action
def Take_Action(state, action):
    row, col = state
    if action == 0:     # Di chuyển xuống
        row = min(row + 1, grid_size - 1)
    elif action == 1:   # Di chuyển lên
        row = max(row - 1, 0)
    elif action == 2:   # Di chuyển trái
        col = max(col - 1, 0)
    elif action == 3:   # Di chuyển phải
        col = min(col + 1, grid_size - 1)
    
    newState = (row, col)

    # Nếu đến được trạng thái đích, nhận thưởng 100, ngược lại nhận -1
    if(newState == goal_state):
        reward = 100
    elif newState in obstacle:
        reward = -2
    elif newState in speed:
        reward = 2
    else:
        reward = -1

    return newState, reward

# -------------------------------------------------------------
# 3. XÂY DƯNG HÀM ĐÁNH GIÁ CHÍNH SÁCH (Policy Evaluation)
# -------------------------------------------------------------
def Policy_Evaluation(policy):
    global V
    while True:
        V_new = np.copy(V)
        for i in range(grid_size - 1, -1, -1):
            for j in range(grid_size - 1, -1, -1):
                state = (i, j)
                # Ở trạng thái đích, coi V(goal)=0 (terminal state)
                if state == goal_state:
                    V_new[i, j] = 0
                    continue
                action = policy[i, j]
                newState, reward = Take_Action(state, action)
                V_new[i, j] = reward + gamma * V[newState]
                
        # Kiểm tra điều kiện hội tụ
        if np.max(abs(V_new - V)) < 1e-6:
            break
        V = V_new

# -------------------------------------------------------------
# 4. XÂY DỰNG HÀM CẢI THIỆN CHÍNH SÁCH (Policy Improvement)
# -------------------------------------------------------------
def Policy_Improvement():
    global policy, V
    policy_stable = True
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            # Nếu state là trạng thái đích, không cần cập nhật lại chính sách
            if state == goal_state:
                continue
            # Tính giá trị cho mỗi hành động có thể thực hiện
            old_action = policy[i, j]
            action_value = []
            for action in range(4):
                newState, reward = Take_Action(state, action)
                action_value.append(reward + gamma * V[newState])
            policy[i, j] = np.argmax(action_value)
            if old_action != policy[i, j]:
                policy_stable = False
    return policy_stable

# -------------------------------------------------------------
# 5. CHẠY CHÍNH SÁCH (Policy Iteration)
# -------------------------------------------------------------
def Policy_Iteration():
    i = 0
    while True:
        i += 1
        print("Iterate {}: \n".format(i))
        print("Policy: \n", policy)
        Policy_Evaluation(policy)
        if Policy_Improvement():
            break

    # Đánh dấu trạng thái đích trong chính sá2ch (ví dụ: -1 để nhận biết)
    policy[goal_state] = -1

# -------------------------------------------------------------
# 6. CHẠY CHƯƠNG TRÌNH
# -------------------------------------------------------------
Policy_Iteration()
print("Chính sách tối ưu:")
print(policy)
print("Hàm giá trị tối ưu:")
print(V)

# ---------------------------
# 7. MÔ PHỎNG ĐƯỜNG ĐI TỐI ƯU
# ---------------------------
def simulate_optimal_path(policy, start, goal):
    """
    Mô phỏng đường đi từ start đến goal theo chính sách policy.
    Tránh vòng lặp vô hạn bằng cách đặt giới hạn số bước.
    """
    path = [start]
    state = start
    max_steps = grid_size * grid_size  # giới hạn bước di chuyển an toàn
    steps = 0

    while state != goal and steps < max_steps:
        # Nếu ở trạng thái đích, thoát vòng lặp
        if state == goal:
            break

        # Nếu ở trạng thái không thể di chuyển (do vòng lặp hay chính sách k hợp lý) thì dừng lại
        action = policy[state]
        # Nếu action = -1 (đánh dấu trạng thái đích) thì thoát
        if action == -1:
            break
        new_state, _ = Take_Action(state, action)
        path.append(new_state)
        state = new_state
        steps += 1
        # print("Bước {}: {} \n".format(steps, state))

    return path

# Mô phỏng đường đi từ (0,0) đến (4,4)
optimal_path = simulate_optimal_path(policy, (0, 0), goal_state)
print("\nĐường đi tối ưu từ (0,0) đến (4,4):")
print(optimal_path)
