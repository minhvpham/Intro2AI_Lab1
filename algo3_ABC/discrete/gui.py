import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from algo3_ABC
from algo3_ABC.discrete.main import (
    two_opt_mutation,
    calculate_path_distance,
    greedy_two_opt,
    nearest_neighbor_path,
)

# Ensure project root is importable for utils
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.tsp import create_cities


class ABC_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ABC TSP Solver - Interactive Visualization")
        self.root.geometry("1400x900")

        # Variables
        self.city_coordinates = None
        self.is_running = False
        self.is_paused = False
        self.current_iteration = 0
        self.animation_speed = 0.2

        # ABC state
        self.foods = None            # (pop, D) int arrays of permutations
        self.fitness = None          # (pop,) costs
        self.trial = None
        self.best_path = None
        self.best_length = np.inf

        # Store iteration snapshots for inspection
        self.iteration_data = []  # list of dicts: {'foods', 'fitness', 'best_path', 'best_length'}

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        # ===== Control Panel =====
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Number of cities input
        ttk.Label(control_frame, text="Number of Cities:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.n_cities_var = tk.StringVar(value="30")
        ttk.Entry(control_frame, textvariable=self.n_cities_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        # Generate cities button
        self.generate_btn = ttk.Button(control_frame, text="Generate Random Cities", command=self.generate_cities)
        self.generate_btn.grid(row=0, column=2, padx=10, pady=5)

        # ABC Parameters
        ttk.Label(control_frame, text="Population (Food Sources):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.pop_var = tk.StringVar(value="40")
        ttk.Entry(control_frame, textvariable=self.pop_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Iterations:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.n_iterations_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.n_iterations_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Limit (scout):").grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.limit_var = tk.StringVar(value="40")
        ttk.Entry(control_frame, textvariable=self.limit_var, width=10).grid(row=1, column=4, padx=5, pady=5)

        ttk.Label(control_frame, text="Show population sample:").grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.sample_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.sample_var, width=10).grid(row=2, column=4, padx=5, pady=5)

        # Run / Pause / Stop buttons
        self.run_btn = ttk.Button(control_frame, text="▶ Run ABC", command=self.start_abc, state="disabled")
        self.run_btn.grid(row=0, column=5, padx=10, pady=5)

        self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", command=self.toggle_pause, state="disabled")
        self.pause_btn.grid(row=0, column=6, padx=10, pady=5)

        self.stop_btn = ttk.Button(control_frame, text="⬛ Stop", command=self.stop_abc, state="disabled")
        self.stop_btn.grid(row=0, column=7, padx=10, pady=5)

        # ===== Inspection Panel =====
        inspect_frame = ttk.LabelFrame(self.root, text="Inspection Panel - View Food Source Path", padding=10)
        inspect_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        ttk.Label(inspect_frame, text="Iteration: ").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.iteration_var = tk.StringVar(value="1")
        self.iteration_entry = ttk.Entry(inspect_frame, textvariable=self.iteration_var, width=10)
        self.iteration_entry.grid(row=0, column=1, padx=5, pady=5)
        self.iteration_label = ttk.Label(inspect_frame, text="(Max: 0)")
        self.iteration_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(inspect_frame, text="Food Index:").grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.food_var = tk.StringVar(value="0")
        self.food_entry = ttk.Entry(inspect_frame, textvariable=self.food_var, width=10)
        self.food_entry.grid(row=0, column=4, padx=5, pady=5)
        self.food_label = ttk.Label(inspect_frame, text="(Max: 0)")
        self.food_label.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        self.inspect_btn = ttk.Button(inspect_frame, text="🔍 Show Food Path", command=self.show_food_path, state="disabled")
        self.inspect_btn.grid(row=0, column=6, columnspan=2, padx=10, pady=5)

        # ===== Status Bar =====
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.status_label = ttk.Label(status_frame, text="Status: Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.progress_label = ttk.Label(status_frame, text="Iteration: 0/0 | Best Length: N/A", relief=tk.SUNKEN, anchor=tk.E, width=40)
        self.progress_label.pack(side=tk.RIGHT, padx=5)

        # ===== Visualization Area =====
        viz_frame = ttk.Frame(self.root)
        viz_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 7))
        self.ax_paths = self.fig.add_subplot(121)
        self.ax_stats = self.fig.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize plots
        self.ax_paths.set_title("City Map (Generate cities to start)")
        self.ax_paths.grid(True, alpha=0.3)
        self.ax_stats.set_title("Population Fitness (No data yet)")

        self.canvas.draw()

    def generate_cities(self):
        try:
            n_cities = int(self.n_cities_var.get())
            if n_cities < 3:
                messagebox.showerror("Error", "Number of cities must be at least 3")
                return
            if n_cities > 200:
                messagebox.showwarning("Warning", "Large number of cities may slow down the visualization")

            self.city_coordinates = create_cities(n_cities, map_size=300, seed=int(time.time()) % 10000)
            self.status_label.config(text=f"Status: Generated {n_cities} cities")
            self.run_btn.config(state="normal")

            # Display cities
            self.ax_paths.clear()
            self.ax_paths.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], 
                                 c='blue', s=100, zorder=5, edgecolors='darkblue', linewidth=2)

            # Add city numbers
            for i, (x, y) in enumerate(self.city_coordinates):
                self.ax_paths.annotate(str(i), (x, y), fontsize=8, fontweight='bold',
                                      ha='center', va='center', color='white',
                                      bbox=dict(boxstyle='circle,pad=0.3', facecolor='blue', 
                                               edgecolor='darkblue', alpha=0.8))

            self.ax_paths.set_title(f"City Map ({n_cities} cities)")
            self.ax_paths.set_xlabel("X Coordinate")
            self.ax_paths.set_ylabel("Y Coordinate")
            self.ax_paths.grid(True, alpha=0.3)

            self.ax_stats.clear()
            self.ax_stats.set_title("Population Fitness (No data yet)")

            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for cities")

    def start_abc(self):
        if self.city_coordinates is None:
            messagebox.showerror("Error", "Please generate cities first")
            return

        try:
            pop = int(self.pop_var.get())
            n_iterations = int(self.n_iterations_var.get())
            limit = int(self.limit_var.get())

            if pop < 2 or n_iterations < 1:
                messagebox.showerror("Error", "Population and iterations must be positive")
                return

            # Initialize ABC population (foods) as random permutations
            D = len(self.city_coordinates)
            self.foods = np.array([np.random.permutation(D) for _ in range(pop)])
            dist_matrix = np.zeros((D, D))
            for i in range(D):
                for j in range(D):
                    dist_matrix[i, j] = np.linalg.norm(self.city_coordinates[i] - self.city_coordinates[j])

            self.fitness = np.array([calculate_path_distance(p, dist_matrix) for p in self.foods])
            self.trial = np.zeros(pop, dtype=int)
            self.best_path = self.foods[np.argmin(self.fitness)].copy()
            self.best_length = float(np.min(self.fitness))

            # Clear previous iteration data
            self.iteration_data = []

            # Update UI state
            self.is_running = True
            self.is_paused = False
            self.current_iteration = 0
            self.run_btn.config(state="disabled")
            self.generate_btn.config(state="disabled")
            self.pause_btn.config(state="normal", text="⏸ Pause")
            self.stop_btn.config(state="normal")
            self.inspect_btn.config(state="disabled")
            self.status_label.config(text="Status: Running ABC...")

            # Run in thread with access to dist_matrix and params
            thread = threading.Thread(target=self.run_abc_iterations, args=(dist_matrix, pop, D, n_iterations, limit), daemon=True)
            thread.start()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")

    def toggle_pause(self):
        """Toggle between pause and resume."""
        if self.is_paused:
            # Resume
            self.is_paused = False
            self.pause_btn.config(text="⏸ Pause")
            self.status_label.config(text="Status: Running ABC...")
        else:
            # Pause
            self.is_paused = True
            self.pause_btn.config(text="▶ Resume")
            self.status_label.config(text="Status: Paused (Click Resume to continue)")

    def stop_abc(self):
        self.is_running = False
        self.is_paused = False
        self.status_label.config(text="Status: Stopped by user")
        self.run_btn.config(state="normal")
        self.generate_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸ Pause")
        self.stop_btn.config(state="disabled")

    def run_abc_iterations(self, dist_matrix, N, D, MaxGen, limit):
        for gen in range(1, MaxGen + 1):
            if not self.is_running:
                break

            # Wait if paused
            while self.is_paused and self.is_running:
                time.sleep(0.1)

            if not self.is_running:
                break

            self.current_iteration = gen

            # Employed bees: local mutation (2-opt)
            for i in range(N):
                new_path = two_opt_mutation(self.foods[i])
                new_fit = calculate_path_distance(new_path, dist_matrix)
                if new_fit < self.fitness[i]:
                    self.foods[i] = new_path
                    self.fitness[i] = new_fit
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1

            # Onlooker bees: probabilistic selection
            inv_fit = 1.0 / (1.0 + self.fitness)
            prob = inv_fit / np.sum(inv_fit)
            for _ in range(N):
                i = np.random.choice(N, p=prob)
                new_path = two_opt_mutation(self.foods[i])
                new_fit = calculate_path_distance(new_path, dist_matrix)
                if new_fit < self.fitness[i]:
                    self.foods[i] = new_path
                    self.fitness[i] = new_fit
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1

            # Memetic intensification on current best
            cur_best_idx = np.argmin(self.fitness)
            if self.trial[cur_best_idx] == 0:
                opt_path, opt_fit = greedy_two_opt(self.foods[cur_best_idx], dist_matrix)
                if opt_fit < self.fitness[cur_best_idx]:
                    self.foods[cur_best_idx] = opt_path
                    self.fitness[cur_best_idx] = opt_fit

            # Scout phase: replace abandoned sources
            scouts = np.where(self.trial > limit)[0]
            for i in scouts:
                self.foods[i] = nearest_neighbor_path(dist_matrix, D)
                self.fitness[i] = calculate_path_distance(self.foods[i], dist_matrix)
                self.trial[i] = 0

            # Update global best
            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_length:
                self.best_length = float(self.fitness[min_idx])
                self.best_path = self.foods[min_idx].copy()

            # Record snapshot
            snapshot = {
                'iteration': self.current_iteration,
                'foods': self.foods.copy(),
                'fitness': self.fitness.copy(),
                'best_path': self.best_path.copy(),
                'best_length': self.best_length
            }
            self.iteration_data.append(snapshot)

            # Update visualization
            sample_n = min(int(self.sample_var.get()), N)
            self.root.after(0, self.update_visualization, snapshot, sample_n)
            time.sleep(self.animation_speed)

            # Update progress
            self.root.after(0, self.update_progress)

        # Finished
        if self.is_running:
            self.root.after(0, self.abc_finished)

    def update_visualization(self, snapshot, sample_n):
        """Draw left: sample of population paths + best; right: fitness bar chart."""
        foods = snapshot['foods']
        fitness = snapshot['fitness']
        best_path = snapshot['best_path']
        best_length = snapshot['best_length']

        D = len(self.city_coordinates)

        self.ax_paths.clear()
        self.ax_stats.clear()

        # Left: sample population paths
        n_show = min(sample_n, len(foods))
        idxs = np.random.choice(len(foods), size=n_show, replace=False)
        colors = plt.cm.tab20(np.linspace(0, 1, n_show))
        for k, idx in enumerate(idxs):
            p = foods[idx]
            coords = self.city_coordinates[p, :]
            coords = np.vstack([coords, coords[0]])
            self.ax_paths.plot(coords[:, 0], coords[:, 1], color=colors[k], alpha=0.4, linewidth=1)

        # Best path
        best_coords = self.city_coordinates[best_path, :]
        best_coords = np.vstack([best_coords, best_coords[0]])
        self.ax_paths.plot(best_coords[:, 0], best_coords[:, 1], color='red', linewidth=2.5, label=f'Best L={best_length:.2f}')
        self.ax_paths.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], c='blue', s=80, zorder=5)
        for i, (x, y) in enumerate(self.city_coordinates):
            self.ax_paths.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', color='white',
                                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='blue', edgecolor='darkblue'))

        self.ax_paths.set_title(f'Iteration {snapshot["iteration"]} - Sample population paths')
        self.ax_paths.set_xlabel('X')
        self.ax_paths.set_ylabel('Y')
        self.ax_paths.grid(True, alpha=0.3)
        self.ax_paths.legend()

        # Right: fitness bar chart (sorted)
        sorted_idx = np.argsort(fitness)
        sorted_fit = fitness[sorted_idx]
        self.ax_stats.bar(range(len(sorted_fit)), sorted_fit, color='C0', alpha=0.7)
        self.ax_stats.set_title('Population Fitness (lower is better)')
        self.ax_stats.set_xlabel('Food source (sorted)')
        self.ax_stats.set_ylabel('Path length')
        self.ax_stats.axhline(best_length, color='red', linestyle='--', label='Best')
        self.ax_stats.legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def update_progress(self):
        total_iters = int(self.n_iterations_var.get())
        self.progress_label.config(text=f"Iteration: {self.current_iteration}/{total_iters} | Best Length: {self.best_length:.2f}")
        # Update inspection labels
        self.iteration_label.config(text=f"(Max: {len(self.iteration_data)})")
        self.food_label.config(text=f"(Max: {len(self.foods) - 1 if self.foods is not None else 0})")

    def abc_finished(self):
        self.is_running = False
        self.is_paused = False
        self.run_btn.config(state="normal")
        self.generate_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸ Pause")
        self.stop_btn.config(state="disabled")
        self.inspect_btn.config(state="normal")

        self.status_label.config(text=f"Status: Completed! Best length: {self.best_length:.2f}")
        messagebox.showinfo("ABC Completed", f"Algorithm finished!\nBest length: {self.best_length:.2f}\nTotal iterations: {len(self.iteration_data)}")

    def show_food_path(self):
        if not self.iteration_data:
            messagebox.showerror("Error", "No iteration data available. Please run the algorithm first.")
            return
        try:
            iteration_num = int(self.iteration_var.get())
            food_idx = int(self.food_var.get())
            iter_idx = iteration_num - 1
            if iter_idx < 0 or iter_idx >= len(self.iteration_data):
                messagebox.showerror("Error", f"Iteration must be between 1 and {len(self.iteration_data)}")
                return
            snapshot = self.iteration_data[iter_idx]
            if food_idx < 0 or food_idx >= len(snapshot['foods']):
                messagebox.showerror("Error", f"Food index must be between 0 and {len(snapshot['foods']) - 1}")
                return
            path = snapshot['foods'][food_idx]
            length = snapshot['fitness'][food_idx]
            self.visualize_food_path(path, length, iter_idx + 1, food_idx)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for iteration and food index")

    def visualize_food_path(self, path, length, iteration_num, food_idx):
        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        coords = self.city_coordinates[path, :]
        coords = np.vstack([coords, coords[0]])
        ax1.plot(coords[:, 0], coords[:, 1], '-o', color='C1')
        ax1.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], c='blue', s=80, zorder=5)
        for i, (x, y) in enumerate(self.city_coordinates):
            ax1.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', color='white',
                         bbox=dict(boxstyle='circle,pad=0.3', facecolor='blue', edgecolor='darkblue'))
        ax1.set_title(f'Iteration {iteration_num} - Food {food_idx} (L={length:.2f})')
        ax1.grid(True, alpha=0.3)

        # Show greedy 2-opt improvement
        D = len(self.city_coordinates)
        dist_matrix = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                dist_matrix[i, j] = np.linalg.norm(self.city_coordinates[i] - self.city_coordinates[j])
        opt_path, opt_len = greedy_two_opt(path, dist_matrix)
        ax2.text(0.1, 0.8, f'Original L = {length:.2f}', fontsize=10)
        ax2.text(0.1, 0.6, f'2-opt L = {opt_len:.2f}', fontsize=10)
        ax2.set_axis_off()

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ABC_GUI(root)
    root.mainloop()
