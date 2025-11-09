import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Circle
import tkinter as tk
from tkinter import ttk, messagebox
from ACO import ACO_TSP_Solver
import threading
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.tsp import create_cities


class ACO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ACO TSP Solver - Interactive Visualization")
        self.root.geometry("1600x900")
        
        # Variables
        self.city_coordinates = None
        self.aco_solver = None
        self.is_running = False
        self.is_paused = False
        self.current_iteration = 0
        self.show_animation = False  # Changed default to False
        self.animation_speed = 0.1  # seconds per step
        
        # Store iteration data for inspection
        self.iteration_data = []  # List of iterations with all paths and ant data
        self.selected_ant = 0
        self.selected_iteration = 0
        
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
        
        # ACO Parameters
        ttk.Label(control_frame, text="Number of Ants:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.n_ants_var = tk.StringVar(value="15")
        ttk.Entry(control_frame, textvariable=self.n_ants_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Iterations:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.n_iterations_var = tk.StringVar(value="20")
        ttk.Entry(control_frame, textvariable=self.n_iterations_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Alpha (α):").grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.alpha_var = tk.StringVar(value="1.0")
        ttk.Entry(control_frame, textvariable=self.alpha_var, width=10).grid(row=1, column=4, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Beta (β):").grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.beta_var = tk.StringVar(value="5.0")
        ttk.Entry(control_frame, textvariable=self.beta_var, width=10).grid(row=2, column=4, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Rho (ρ):").grid(row=1, column=5, padx=5, pady=5, sticky="w")
        self.rho_var = tk.StringVar(value="0.5")
        ttk.Entry(control_frame, textvariable=self.rho_var, width=10).grid(row=1, column=6, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Q:").grid(row=2, column=5, padx=5, pady=5, sticky="w")
        self.q_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.q_var, width=10).grid(row=2, column=6, padx=5, pady=5)
        
        # Run button
        self.run_btn = ttk.Button(control_frame, text="▶ Run ACO", command=self.start_aco, state="disabled")
        self.run_btn.grid(row=0, column=3, columnspan=2, padx=10, pady=5)
        
        # Pause/Resume button
        self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", command=self.toggle_pause, state="disabled")
        self.pause_btn.grid(row=0, column=5, padx=10, pady=5)
        
        # Stop button
        self.stop_btn = ttk.Button(control_frame, text="⬛ Stop", command=self.stop_aco, state="disabled")
        self.stop_btn.grid(row=0, column=6, padx=10, pady=5)
        
        # ===== Inspection Panel =====
        inspect_frame = ttk.LabelFrame(self.root, text="Inspection Panel - View Complete Ant Path & Decisions", padding=10)
        inspect_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(inspect_frame, text="Iteration:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.iteration_var = tk.StringVar(value="1")
        self.iteration_entry = ttk.Entry(inspect_frame, textvariable=self.iteration_var, width=10)
        self.iteration_entry.grid(row=0, column=1, padx=5, pady=5)
        self.iteration_label = ttk.Label(inspect_frame, text="(Max: 0)")
        self.iteration_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(inspect_frame, text="Ant:").grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.ant_var = tk.StringVar(value="0")
        self.ant_entry = ttk.Entry(inspect_frame, textvariable=self.ant_var, width=10)
        self.ant_entry.grid(row=0, column=4, padx=5, pady=5)
        self.ant_label = ttk.Label(inspect_frame, text="(Max: 0)")
        self.ant_label.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        
        self.inspect_btn = ttk.Button(inspect_frame, text="🔍 Show Complete Path & Decisions", 
                                     command=self.show_ant_decision, state="disabled")
        self.inspect_btn.grid(row=0, column=6, columnspan=2, padx=10, pady=5)
        
        # ===== Status Bar =====
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.progress_label = ttk.Label(status_frame, text="Iteration: 0/0 | Best Length: N/A", relief=tk.SUNKEN, anchor=tk.E, width=40)
        self.progress_label.pack(side=tk.RIGHT, padx=5)
        
        # Add ant status label
        self.ant_status_label = ttk.Label(status_frame, text="Ants: Ready", relief=tk.SUNKEN, anchor=tk.CENTER, width=50)
        self.ant_status_label.pack(side=tk.RIGHT, padx=5)
        
        # ===== Visualization Area =====
        viz_frame = ttk.Frame(self.root)
        viz_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(16, 7))
        self.ax_paths = self.fig.add_subplot(121)
        self.ax_pheromones = self.fig.add_subplot(122)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.ax_paths.set_title("City Map (Generate cities to start)")
        self.ax_paths.grid(True, alpha=0.3)
        self.ax_pheromones.set_title("Pheromone Levels")
        
        self.canvas.draw()
        
    def generate_cities(self):
        try:
            n_cities = int(self.n_cities_var.get())
            if n_cities < 3:
                messagebox.showerror("Error", "Number of cities must be at least 3")
                return
            if n_cities > 100:
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
            
            self.ax_pheromones.clear()
            self.ax_pheromones.set_title("Pheromone Levels (No data yet)")
            
            self.canvas.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for cities")
    
    def toggle_animation(self):
        self.show_animation = self.animation_var.get()
    
    def start_aco(self):
        if self.city_coordinates is None:
            messagebox.showerror("Error", "Please generate cities first")
            return
        
        try:
            # Get parameters
            n_ants = int(self.n_ants_var.get())
            n_iterations = int(self.n_iterations_var.get())
            alpha = float(self.alpha_var.get())
            beta = float(self.beta_var.get())
            rho = float(self.rho_var.get())
            q = float(self.q_var.get())
            
            # Validate parameters
            if n_ants < 1 or n_iterations < 1:
                messagebox.showerror("Error", "Ants and iterations must be positive")
                return
            
            # Create solver
            self.aco_solver = ACO_TSP_Solver(self.city_coordinates, n_ants, n_iterations,
                                            alpha, beta, rho, q)
            
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
            self.status_label.config(text="Status: Running ACO...")
            
            # Run in separate thread
            thread = threading.Thread(target=self.run_aco_iterations, daemon=True)
            thread.start()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
    
    def toggle_pause(self):
        """Toggle between pause and resume."""
        if self.is_paused:
            # Resume
            self.is_paused = False
            self.pause_btn.config(text="⏸ Pause")
            self.status_label.config(text="Status: Running ACO...")
        else:
            # Pause
            self.is_paused = True
            self.pause_btn.config(text="▶ Resume")
            self.status_label.config(text="Status: Paused (Click Resume to continue)")
    
    def stop_aco(self):
        self.is_running = False
        self.is_paused = False
        self.status_label.config(text="Status: Stopped by user")
        self.run_btn.config(state="normal")
        self.generate_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸ Pause")
        self.stop_btn.config(state="disabled")
    
    def run_aco_iterations(self):
        for iteration in range(self.aco_solver.n_iterations):
            if not self.is_running:
                break
            
            # Wait if paused
            while self.is_paused and self.is_running:
                time.sleep(0.1)
            
            if not self.is_running:
                break
            
            self.current_iteration = iteration + 1
            
            # Build solutions and collect detailed data
            all_paths, all_lengths, ant_decision_data = self.build_solutions_with_data_collection()
            
            # Store iteration data for inspection
            iteration_info = {
                'iteration': self.current_iteration,
                'paths': all_paths,
                'lengths': all_lengths,
                'ant_decisions': ant_decision_data,
                'pheromones': self.aco_solver.pheromones.copy()
            }
            self.iteration_data.append(iteration_info)
            
            # Update visualization (must be done in main thread)
            self.root.after(0, self.update_visualization, all_paths, all_lengths)
            time.sleep(0.3)
            
            # Evaporation
            self.aco_solver.pheromones *= (1.0 - self.aco_solver.rho)
            
            # Deposition
            self.aco_solver._pheromone_deposition(all_paths, all_lengths)
            
            # Update best-so-far
            current_best_length = np.min(all_lengths)
            current_best_path = all_paths[np.argmin(all_lengths)]
            
            self.aco_solver.iteration_best_paths.append(current_best_path)
            self.aco_solver.iteration_best_lengths.append(current_best_length)
            
            if current_best_length < self.aco_solver.best_path_length:
                self.aco_solver.best_path_length = current_best_length
                self.aco_solver.best_path = current_best_path
            
            self.aco_solver.convergence_history.append(self.aco_solver.best_path_length)
            
            # Update progress
            self.root.after(0, self.update_progress)
        
        # Finished
        if self.is_running:
            self.root.after(0, self.aco_finished)
    
    def build_solutions_with_data_collection(self):
        """Build solutions while collecting detailed decision data for each ant."""
        all_paths = []
        all_lengths = []
        ant_decision_data = []
        
        for ant_idx in range(self.aco_solver.n_ants):
            path = []
            visited = np.zeros(self.aco_solver.n_cities, dtype=bool)
            decisions = []  # Store each decision point
            
            current_city = np.random.randint(0, self.aco_solver.n_cities)
            path.append(current_city)
            visited[current_city] = True
            
            while len(path) < self.aco_solver.n_cities:
                # Get all possible next cities and their probabilities
                unvisited = np.where(~visited)[0]
                
                tau_values = self.aco_solver.pheromones[current_city, :]
                eta_values = self.aco_solver.eta[current_city, :]
                
                tau_pow = np.power(tau_values, self.aco_solver.alpha)
                eta_pow = np.power(eta_values, self.aco_solver.beta)
                
                probabilities = tau_pow * eta_pow
                probabilities[visited] = 0
                
                sum_probs = np.sum(probabilities)
                if sum_probs > 0:
                    probabilities /= sum_probs
                
                # Select next city
                next_city = self.aco_solver._select_next_city(current_city, visited)
                
                # Store decision information
                decision_info = {
                    'from_city': current_city,
                    'to_city': next_city,
                    'unvisited_cities': unvisited.tolist(),
                    'probabilities': {city: probabilities[city] for city in unvisited},
                    'pheromones': {city: tau_values[city] for city in unvisited},
                    'heuristics': {city: eta_values[city] for city in unvisited},
                    'distances': {city: self.aco_solver.distances[current_city, city] for city in unvisited}
                }
                decisions.append(decision_info)
                
                path.append(next_city)
                visited[next_city] = True
                current_city = next_city
            
            path.append(path[0])  # Return to start
            length = self.aco_solver._calculate_path_length(path)
            
            all_paths.append(path)
            all_lengths.append(length)
            ant_decision_data.append({'path': path, 'length': length, 'decisions': decisions})
        
        return all_paths, all_lengths, ant_decision_data

    def update_visualization(self, all_paths, all_lengths):
        # Clear previous plots
        self.ax_paths.clear()
        self.ax_pheromones.clear()
        
        # ===== Left plot: Ant paths =====
        # Plot each ant's path with different colors (FIRST - so they're in background)
        n_ants_to_show = min(10, len(all_paths))  # Limit displayed ants for clarity
        colors = plt.cm.tab20(np.linspace(0, 1, n_ants_to_show))
        
        for ant_idx in range(n_ants_to_show):
            path = all_paths[ant_idx]
            length = all_lengths[ant_idx]
            path_coords = self.city_coordinates[path, :]
            self.ax_paths.plot(path_coords[:, 0], path_coords[:, 1], 
                              color=colors[ant_idx], linewidth=1.5, alpha=0.4, zorder=1)
        
        # Plot global best if available
        if self.aco_solver.best_path is not None:
            best_global_coords = self.city_coordinates[self.aco_solver.best_path, :]
            self.ax_paths.plot(best_global_coords[:, 0], best_global_coords[:, 1], 
                              c='green', linewidth=2, linestyle='--', alpha=0.7,
                              label=f"Global best (L: {self.aco_solver.best_path_length:.2f})", 
                              zorder=2)
        
        # Highlight best path in this iteration
        best_idx = np.argmin(all_lengths)
        best_path = all_paths[best_idx]
        best_length = all_lengths[best_idx]
        
        if len(best_path) > 1:
            path_coords = self.city_coordinates[best_path, :]
            self.ax_paths.plot(path_coords[:, 0], path_coords[:, 1], c='red', 
                              linewidth=3, linestyle='-', marker='o', markersize=6, 
                              label=f"Best this iteration (L: {best_length:.2f})", zorder=3)
            self.ax_paths.scatter(self.city_coordinates[best_path[0], 0], 
                                 self.city_coordinates[best_path[0], 1],
                                 c='gold', s=300, marker='*', zorder=4, 
                                 edgecolors='orange', linewidth=2, label="Start")
        
        # Draw cities and labels LAST (so they're on top)
        self.ax_paths.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], 
                             c='blue', s=100, zorder=5, edgecolors='darkblue', linewidth=2, 
                             label="Cities")
        
        # Add city numbers on top of everything
        for i, (x, y) in enumerate(self.city_coordinates):
            self.ax_paths.annotate(str(i), (x, y), fontsize=9, fontweight='bold',
                                  ha='center', va='center', color='white',
                                  bbox=dict(boxstyle='circle,pad=0.3', facecolor='blue', 
                                           edgecolor='white', linewidth=1.5, alpha=0.95), zorder=100)
        
        self.ax_paths.set_title(f"Iteration {self.current_iteration} - Ant Paths", 
                               fontsize=12, fontweight='bold')
        self.ax_paths.set_xlabel("X Coordinate")
        self.ax_paths.set_ylabel("Y Coordinate")
        self.ax_paths.grid(True, alpha=0.3)
        self.ax_paths.legend(fontsize=9, loc='upper right')
        
        # ===== Right plot: Pheromone heatmap =====
        pheromone_normalized = self.aco_solver.pheromones.copy()
        pheromone_max = np.max(pheromone_normalized)
        if pheromone_max > 0:
            pheromone_normalized = pheromone_normalized / pheromone_max
        
        im = self.ax_pheromones.imshow(pheromone_normalized, cmap='YlOrRd', 
                                       aspect='auto', vmin=0, vmax=1)
        self.ax_pheromones.set_title("Pheromone Intensity on Edges", 
                                     fontsize=12, fontweight='bold')
        self.ax_pheromones.set_xlabel("To City")
        self.ax_pheromones.set_ylabel("From City")
        
        # Add colorbar if not exists
        if not hasattr(self, 'colorbar'):
            self.colorbar = self.fig.colorbar(im, ax=self.ax_pheromones, 
                                             label="Pheromone Level (Normalized)")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_progress(self):
        self.progress_label.config(
            text=f"Iteration: {self.current_iteration}/{self.aco_solver.n_iterations} | "
                 f"Best Length: {self.aco_solver.best_path_length:.2f}"
        )
    
    def aco_finished(self):
        self.is_running = False
        self.is_paused = False
        self.run_btn.config(state="normal")
        self.generate_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸ Pause")
        self.stop_btn.config(state="disabled")
        self.inspect_btn.config(state="normal")
        
        # Update labels with max values
        n_iterations = len(self.iteration_data)
        n_ants = self.aco_solver.n_ants
        
        self.iteration_label.config(text=f"(Max: {n_iterations})")
        self.ant_label.config(text=f"(Max: {n_ants - 1})")
        
        # Set default values
        self.iteration_var.set("1")
        self.ant_var.set("0")
        
        self.status_label.config(text=f"Status: Completed! Best path length: {self.aco_solver.best_path_length:.2f} - Enter values to inspect complete ant paths")
        messagebox.showinfo("ACO Completed", 
                           f"Algorithm finished!\n\nBest path length: {self.aco_solver.best_path_length:.2f}\n"
                           f"Total iterations: {self.aco_solver.n_iterations}\n\n"
                           f"Use the Inspection Panel to examine ant decision-making!")
    
    def show_ant_decision(self):
        """Show complete path with all decisions for selected ant."""
        if not self.iteration_data:
            messagebox.showerror("Error", "No iteration data available. Please run the algorithm first.")
            return
        
        try:
            # Get values from text inputs
            iteration_num = int(self.iteration_var.get())
            ant_num = int(self.ant_var.get())
            
            # Convert to 0-indexed for iteration
            iteration_idx = iteration_num - 1
            ant_idx = ant_num
            
            # Validate ranges
            n_iterations = len(self.iteration_data)
            n_ants = self.aco_solver.n_ants
            
            if iteration_idx < 0 or iteration_idx >= n_iterations:
                messagebox.showerror("Error", f"Iteration must be between 1 and {n_iterations}")
                return
            
            if ant_idx < 0 or ant_idx >= n_ants:
                messagebox.showerror("Error", f"Ant must be between 0 and {n_ants - 1}")
                return
            
            iteration_info = self.iteration_data[iteration_idx]
            ant_data = iteration_info['ant_decisions'][ant_idx]
            
            # Visualize the complete path
            self.visualize_complete_ant_path(ant_data, iteration_info, iteration_num, ant_num)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for Iteration and Ant")
    
    def visualize_complete_ant_path(self, ant_data, iteration_info, iteration_num, ant_num):
        """Visualize complete ant path with all decisions and pheromone/distance matrix."""
        # Clear the figure completely to remove old colorbars
        self.fig.clear()
        
        # Recreate the subplots
        self.ax_paths = self.fig.add_subplot(121)
        self.ax_pheromones = self.fig.add_subplot(122)
        
        path = ant_data['path']
        decisions = ant_data['decisions']
        pheromones = iteration_info['pheromones']
        
        # ===== Left plot: Complete path with all decision probabilities =====
        
        # Draw all cities (light gray)
        self.ax_paths.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], 
                             c='lightgray', s=150, zorder=2, edgecolors='gray', linewidth=2, alpha=0.5)
        
        # Draw all alternative edges (red dashed) for each decision
        for decision in decisions:
            from_city = decision['from_city']
            chosen_city = decision['to_city']
            unvisited = decision['unvisited_cities']
            probabilities = decision['probabilities']
            
            x_from, y_from = self.city_coordinates[from_city]
            
            # Draw unchosen options in red
            for city in unvisited:
                if city != chosen_city:
                    x_to, y_to = self.city_coordinates[city]
                    prob = probabilities.get(city, 0)
                    
                    self.ax_paths.plot([x_from, x_to], [y_from, y_to], 
                                      color='red', linewidth=1.5, alpha=0.3, zorder=4,
                                      linestyle='--')
                    
                    # Add probability label (smaller font for clarity)
                    mid_x, mid_y = (x_from + x_to) / 2, (y_from + y_to) / 2
                    self.ax_paths.annotate(f'{prob:.2f}', (mid_x, mid_y), 
                                          fontsize=6, color='darkred', 
                                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                                   alpha=0.6, edgecolor='red', linewidth=0.5))
        
        # Draw chosen path (green) with probabilities
        for i, decision in enumerate(decisions):
            from_city = decision['from_city']
            to_city = decision['to_city']
            prob = decision['probabilities'].get(to_city, 0)
            
            x_from, y_from = self.city_coordinates[from_city]
            x_to, y_to = self.city_coordinates[to_city]
            
            # Draw chosen edge in green
            self.ax_paths.plot([x_from, x_to], [y_from, y_to], 
                              color='green', linewidth=3, alpha=0.8, zorder=6)
            
            # Add arrow
            dx, dy = x_to - x_from, y_to - y_from
            self.ax_paths.arrow(x_from, y_from, dx * 0.7, dy * 0.7,
                               head_width=2, head_length=1.5, fc='green', ec='darkgreen', 
                               linewidth=2, alpha=0.8, zorder=7)
            
            # Add probability label on chosen path
            mid_x, mid_y = (x_from + x_to) / 2, (y_from + y_to) / 2
            self.ax_paths.annotate(f'P={prob:.3f}', (mid_x, mid_y), 
                                  fontsize=7, color='white', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='darkgreen', 
                                           edgecolor='lime', linewidth=1.5, alpha=0.9), zorder=8)
        
        # Highlight start city
        start_city = path[0]
        x_start, y_start = self.city_coordinates[start_city]
        self.ax_paths.scatter(x_start, y_start, c='gold', s=600, marker='*', 
                             zorder=10, edgecolors='orange', linewidth=3, 
                             label=f'Start (City {start_city})')
        
        # Add city numbers ON TOP of everything (highest z-order)
        for i, (x, y) in enumerate(self.city_coordinates):
            self.ax_paths.annotate(str(i), (x, y), fontsize=8, fontweight='bold',
                                  ha='center', va='center', color='black',
                                  bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                                           edgecolor='darkblue', linewidth=2, alpha=0.95), zorder=20)
        
        path_length = ant_data['length']
        self.ax_paths.set_title(
            f"Iteration {iteration_num} - Ant {ant_num} - Complete Path\n"
            f"Path Length: {path_length:.2f} | Green: Chosen | Red: Alternatives",
            fontsize=11, fontweight='bold')
        self.ax_paths.set_xlabel("X Coordinate")
        self.ax_paths.set_ylabel("Y Coordinate")
        self.ax_paths.grid(True, alpha=0.3)
        self.ax_paths.legend(fontsize=9, loc='upper right')
        
        # ===== Right plot: Pheromone and Distance Matrix =====
        
        n_cities = len(self.city_coordinates)
        
        # Create combined matrix visualization
        # We'll show pheromone as heatmap background and annotate with both pheromone and distance
        
        # Use pheromone values for the heatmap
        matrix_data = pheromones.copy()
        
        # Create heatmap
        im = self.ax_pheromones.imshow(matrix_data, cmap='YlOrRd', aspect='auto', 
                                        vmin=0, vmax=np.max(pheromones))
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax_pheromones, fraction=0.046, pad=0.04)
        cbar.set_label('Pheromone Level', rotation=270, labelpad=15, fontsize=9)
        
        # Calculate distances for annotation
        distances = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distances[i, j] = np.linalg.norm(
                        self.city_coordinates[i] - self.city_coordinates[j]
                    )
        
        # Annotate each cell with pheromone and distance
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    phero = pheromones[i, j]
                    dist = distances[i, j]
                    
                    # Check if this edge is in the ant's path
                    edge_in_path = False
                    for k in range(len(path) - 1):
                        if path[k] == i and path[k + 1] == j:
                            edge_in_path = True
                            break
                    
                    # Color text based on whether edge is in path
                    text_color = 'lime' if edge_in_path else 'black'
                    text_weight = 'bold' if edge_in_path else 'normal'
                    
                    self.ax_pheromones.text(j, i, f'τ:{phero:.2f}\nd:{dist:.1f}',
                                           ha='center', va='center',
                                           fontsize=6, color=text_color,
                                           fontweight=text_weight)
                else:
                    # Diagonal - no self-loops
                    self.ax_pheromones.text(j, i, '-',
                                           ha='center', va='center',
                                           fontsize=10, color='gray')
        
        # Set ticks and labels
        self.ax_pheromones.set_xticks(np.arange(n_cities))
        self.ax_pheromones.set_yticks(np.arange(n_cities))
        self.ax_pheromones.set_xticklabels(np.arange(n_cities), fontsize=8)
        self.ax_pheromones.set_yticklabels(np.arange(n_cities), fontsize=8)
        
        # Add grid
        self.ax_pheromones.set_xticks(np.arange(n_cities) - 0.5, minor=True)
        self.ax_pheromones.set_yticks(np.arange(n_cities) - 0.5, minor=True)
        self.ax_pheromones.grid(which='minor', color='white', linestyle='-', linewidth=2)
        
        self.ax_pheromones.set_title(
            f"Pheromone (τ) & Distance (d) Matrix\n"
            f"Iteration {iteration_num} | Ant {ant_num}'s edges in green",
            fontsize=10, fontweight='bold')
        self.ax_pheromones.set_xlabel("To City", fontsize=9)
        self.ax_pheromones.set_ylabel("From City", fontsize=9)
        
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ACO_GUI(root)
    root.mainloop()
