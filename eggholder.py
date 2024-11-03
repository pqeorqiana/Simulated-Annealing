import random
import matplotlib.pyplot as plt
import numpy as np


class EggholderOptimization:
    def __init__(self, temp=2000, cooling_rate=0.97, iterations=500, local_searches=20,
                 multiplier=[1.0, 0.1], lower_bound=-512, upper_bound=512):
        self.temperature_0 = temp
        self.temperature = temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.local_searches = local_searches
        self.multiplier = multiplier
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.history = []
        self.acceptance_probability_history = []
        self.best_solution = None
        self.best_cost = float('inf')

    def starting_point(self):
        x = random.uniform(self.lower_bound, self.upper_bound)
        y = random.uniform(self.lower_bound, self.upper_bound)
        return x, y

    def neighbour(self, x, y, multiplier=1.0):
        new_x = x + random.uniform(-1, 1) * multiplier * (self.temperature / self.temperature_0) * 100
        new_y = y + random.uniform(-1, 1) * multiplier * (self.temperature / self.temperature_0) * 100
        # Asigură că punctele rămân în limitele definite
        new_x = max(min(new_x, self.upper_bound), self.lower_bound)
        new_y = max(min(new_y, self.upper_bound), self.lower_bound)
        return new_x, new_y

    def eggholder_function(self, x, y):
        """
        Implementarea funcției Eggholder
        Minimul cunoscut la f(512, 404.2319) ≈ -959.6407
        """
        term1 = -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47))))
        term2 = -x * np.sin(np.sqrt(abs(x - (y + 47))))
        return term1 + term2

    def acceptance_probability(self, old_cost, new_cost):
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / self.temperature)

    def optimize(self):
        # Soluția inițială
        x, y = self.starting_point()
        current_cost = self.eggholder_function(x, y)
        self.history.append((x, y))

        # Urmărește cea mai bună soluție
        self.best_solution = (x, y)
        self.best_cost = current_cost

        for iteration in range(self.iterations):
            # Căutare globală
            new_x, new_y = self.neighbour(x, y, self.multiplier[0])
            new_cost = self.eggholder_function(new_x, new_y)

            acc_prob = self.acceptance_probability(current_cost, new_cost)
            if acc_prob > random.random():
                x, y = new_x, new_y
                current_cost = new_cost
                self.acceptance_probability_history.append(acc_prob)
                self.history.append((x, y))

                # Actualizează cea mai bună soluție dacă este necesar
                if current_cost < self.best_cost:
                    self.best_solution = (x, y)
                    self.best_cost = current_cost

            # Căutări locale
            for _ in range(self.local_searches):
                new_x, new_y = self.neighbour(x, y, self.multiplier[1])
                new_cost = self.eggholder_function(new_x, new_y)

                acc_prob = self.acceptance_probability(current_cost, new_cost)
                if acc_prob > random.random():
                    x, y = new_x, new_y
                    current_cost = new_cost
                    self.acceptance_probability_history.append(acc_prob)
                    self.history.append((x, y))

                    if current_cost < self.best_cost:
                        self.best_solution = (x, y)
                        self.best_cost = current_cost

            self.temperature *= self.cooling_rate

        return self.best_solution[0], self.best_solution[1], self.best_cost

    def plot(self):
        # Creează grila pentru reprezentare grafică
        x = np.linspace(self.lower_bound, self.upper_bound, 200)
        y = np.linspace(self.lower_bound, self.upper_bound, 200)
        X, Y = np.meshgrid(x, y)
        Z = self.eggholder_function(X, Y)

        # Creează figura cu subploturi
        fig = plt.figure(figsize=(15, 8))

        # Plot suprafață 3D
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        ax.set_title('Traseul Optimizării Funcției Eggholder')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')

        # Plotează traseul optimizării
        hx, hy = zip(*self.history)
        hz = [self.eggholder_function(x, y) for x, y in self.history]
        ax.plot(hx, hy, hz, color='red', marker='.', markersize=1, linewidth=1, label='Traseu Optimizare')

        # Plotează cel mai bun punct
        ax.scatter(*self.best_solution, self.best_cost, color='red', s=100, label='Soluția Optimă')

        # Adaugă text cu rezultate
        ax.text2D(0.02, -0.1,
                  f'Soluția optimă:\nx = {self.best_solution[0]:.6f}\ny = {self.best_solution[1]:.6f}\n'
                  f'f(x,y) = {self.best_cost:.6f}\n'
                  f'Minim cunoscut: f(512, 404.2319) ≈ -959.6407',
                  transform=ax.transAxes)

        # Plot convergență
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(hz, color='blue', linewidth=1, label='Valoarea Funcției')
        ax2.set_title('Istoricul Convergenței')
        ax2.set_xlabel('Iterație')
        ax2.set_ylabel('Valoarea Funcției')
        ax2.axhline(y=-959.6407, color='r', linestyle='--', label='Minim Global')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Rulează de mai multe ori pentru a găsi minimul global
    best_result = float('inf')
    best_run = None

    print("Minim global cunoscut: f(512, 404.2319) ≈ -959.6407")

    for run in range(15):
        print(f"\nRularea {run + 1}/5")
        optimizer = EggholderOptimization()
        x, y, cost = optimizer.optimize()
        print(f"Minim găsit la x={x:.6f}, y={y:.6f} cu f(x,y)={cost:.6f}")

        if cost < best_result:
            best_result = cost
            best_run = optimizer

    print(f"\nCel mai bun rezultat din toate rulările: f(x,y)={best_result:.6f}")
    best_run.plot()