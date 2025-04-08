import pyautogui
import time

class MatrixAgent:
    def __init__(self):
        self.matrix = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0]
        ]

        self.state_map = [['E' if cell == 1 else ' ' for cell in row] for row in self.matrix]
        self.current_pos = (4, 1)  # y, x
        self.visited = set()

    def move(self, direction):
        keys = {'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'}
        key = keys[direction]
        for _ in range(2):  # Her kareye ulaşmak için 2 hareket gerekiyor
            pyautogui.keyDown(key)
            time.sleep(0.4)  # hareket süresi azaltıldı
            pyautogui.keyUp(key)
            time.sleep(0.2)  # adımlar arası bekleme
        time.sleep(0.8)  # hareket sonu bekleme - shock face için

    def get_neighbors(self, y, x):
        neighbors = []
        directions = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
        for dy, dx, dir in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(self.matrix) and 0 <= nx < len(self.matrix[0]):
                if self.matrix[ny][nx] == 1 and (ny, nx) not in self.visited:
                    neighbors.append(((ny, nx), dir))
        return neighbors

    def run(self):
        stack = [(self.current_pos, None)]

        while stack:
            (y, x), move_dir = stack.pop()

            if (y, x) in self.visited:
                continue

            if move_dir:
                self.move(move_dir)

            print(f"🟩 Ziyaret edilen kare: ({y}, {x})")
            self.visited.add((y, x))
            self.state_map[y][x] = 'V'

            neighbors = self.get_neighbors(y, x)
            for n in neighbors:
                stack.append(n)

if __name__ == "__main__":
    agent = MatrixAgent()
    agent.run()
