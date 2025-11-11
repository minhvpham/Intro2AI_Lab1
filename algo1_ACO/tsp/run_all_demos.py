"""
Quick Run Script for All ACO Demonstrations
============================================
This script provides an interactive menu to run all three demonstrations
showing the exploration vs. exploitation trade-off in ACO.
"""

import os
import sys

def print_banner():
    print("\n" + "="*70)
    print(" " * 15 + "ACO EXPLORATION vs EXPLOITATION")
    print(" " * 20 + "DEMONSTRATION SUITE")
    print("="*70)
    print("\n🐜 Understanding the Core Trade-off in Metaheuristic Algorithms 🐜\n")


def print_menu():
    print("\n" + "-"*70)
    print("AVAILABLE DEMONSTRATIONS:")
    print("-"*70)
    print("\n[1] 🔥 EXPLOITATION DEMO - Simple Problem (10 cities)")
    print("    → Shows: Fast convergence with high exploitation")
    print("    → Time: ~30 seconds")
    print("    → Settings: High α (2.0), High β (5.0), Low ρ (0.1)")
    
    print("\n[2] 🌊 EXPLORATION DEMO - Complex Problem (30 cities)")
    print("    → Shows: Better quality with high exploration")
    print("    → Time: ~2 minutes")
    print("    → Settings: Low α (0.5), Low β (1.0), High ρ (0.7)")
    print("    → Includes: Direct comparison with exploitation")
    
    print("\n[3] 🔬 PARAMETER SENSITIVITY ANALYSIS (20 cities)")
    print("    → Shows: How α, β, ρ affect performance")
    print("    → Time: ~5-10 minutes")
    print("    → Analysis: 60+ experiments across parameter space")
    print("    → Output: Multiple comprehensive visualizations")
    
    print("\n[4] 🚀 RUN ALL DEMONSTRATIONS")
    print("    → Runs all three in sequence")
    print("    → Time: ~15 minutes total")
    print("    → Perfect for complete understanding")
    
    print("\n[5] 📖 VIEW README")
    print("    → Opens the comprehensive documentation")
    
    print("\n[0] ❌ EXIT")
    print("-"*70)


def run_exploitation_demo():
    print("\n" + "🔥"*35)
    print("STARTING EXPLOITATION DEMO")
    print("🔥"*35 + "\n")
    import exploitation_demo
    exploitation_demo.run_exploitation_demo()


def run_exploration_demo():
    print("\n" + "🌊"*35)
    print("STARTING EXPLORATION DEMO")
    print("🌊"*35 + "\n")
    import exploration_demo
    exploration_demo.run_exploration_demo()


def run_sensitivity_analysis():
    print("\n" + "🔬"*35)
    print("STARTING PARAMETER SENSITIVITY ANALYSIS")
    print("🔬"*35 + "\n")
    import parameter_sensitivity_analysis
    parameter_sensitivity_analysis.run_full_sensitivity_analysis()


def view_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'EXPLORATION_EXPLOITATION_README.md')
    if os.path.exists(readme_path):
        print("\n" + "="*70)
        print("Opening README file...")
        print("="*70)
        
        # Try to open with default viewer
        try:
            if sys.platform == 'win32':
                os.startfile(readme_path)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{readme_path}"')
            else:  # linux
                os.system(f'xdg-open "{readme_path}"')
            print(f"\n✅ README opened: {readme_path}")
        except:
            print(f"\n📄 README location: {readme_path}")
            print("Please open it manually.")
    else:
        print("\n❌ README file not found!")


def main():
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n👉 Enter your choice (0-5): ").strip()
            
            if choice == '0':
                print("\n" + "="*70)
                print("Thank you for exploring ACO optimization!")
                print("🐜 May your algorithms always find the optimal path! 🐜")
                print("="*70 + "\n")
                break
                
            elif choice == '1':
                run_exploitation_demo()
                input("\n✅ Press Enter to return to menu...")
                
            elif choice == '2':
                run_exploration_demo()
                input("\n✅ Press Enter to return to menu...")
                
            elif choice == '3':
                print("\n⚠️  WARNING: This analysis takes 5-10 minutes!")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_sensitivity_analysis()
                    input("\n✅ Press Enter to return to menu...")
                else:
                    print("Analysis cancelled.")
                
            elif choice == '4':
                print("\n" + "🚀"*35)
                print("RUNNING ALL DEMONSTRATIONS")
                print("🚀"*35)
                print("\nThis will take approximately 15 minutes.")
                print("You can stop at any time with Ctrl+C.\n")
                
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    try:
                        print("\n" + "="*70)
                        print("STEP 1/3: EXPLOITATION DEMO")
                        print("="*70)
                        run_exploitation_demo()
                        
                        print("\n" + "="*70)
                        print("STEP 2/3: EXPLORATION DEMO")
                        print("="*70)
                        run_exploration_demo()
                        
                        print("\n" + "="*70)
                        print("STEP 3/3: PARAMETER SENSITIVITY ANALYSIS")
                        print("="*70)
                        run_sensitivity_analysis()
                        
                        print("\n" + "🎉"*35)
                        print("ALL DEMONSTRATIONS COMPLETED!")
                        print("🎉"*35)
                        print("\n✅ All visualizations have been saved to PNG files.")
                        print("✅ Check the current directory for the results.\n")
                        
                    except KeyboardInterrupt:
                        print("\n\n⚠️  Demonstrations interrupted by user.")
                    
                    input("\n✅ Press Enter to return to menu...")
                else:
                    print("Cancelled.")
                
            elif choice == '5':
                view_readme()
                input("\n✅ Press Enter to return to menu...")
                
            else:
                print("\n❌ Invalid choice! Please enter a number between 0 and 5.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            confirm = input("Exit program? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\n👋 Goodbye!\n")
                break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or choose a different option.")
            input("Press Enter to continue...")


def print_quick_info():
    """Print quick info about the demonstrations."""
    print("\n" + "ℹ️ "*35)
    print("QUICK INFO")
    print("ℹ️ "*35)
    print("\n📚 THEORETICAL BACKGROUND:")
    print("   The Exploration vs. Exploitation trade-off is the central")
    print("   challenge in ALL metaheuristic algorithms.")
    print("\n   In ACO, this trade-off is managed through three parameters:")
    print("   • α (alpha) - Pheromone weight: Trust in learned paths")
    print("   • β (beta)  - Heuristic weight: Greediness")
    print("   • ρ (rho)   - Evaporation rate: Forgetting speed")
    
    print("\n🎯 EXPLOITATION (Fast convergence, risk of local optima):")
    print("   High α + High β + Low ρ")
    print("   ➤ Best for: Simple problems, limited iterations")
    
    print("\n🎯 EXPLORATION (Diverse search, better global solutions):")
    print("   Low α + Low β + High ρ")
    print("   ➤ Best for: Complex problems, avoiding local optima")
    
    print("\n💡 THE KEY INSIGHT:")
    print("   These parameters are NOT just 'fine-tuning numbers'.")
    print("   They are the EXPLICIT MECHANISM for controlling the")
    print("   exploration-exploitation trade-off!")
    
    print("\n📊 WHAT YOU'LL LEARN:")
    print("   ✓ Why parameter tuning matters")
    print("   ✓ How to visualize algorithm behavior")
    print("   ✓ When to use exploitation vs exploration")
    print("   ✓ How to find optimal parameter settings")
    
    print("\n" + "="*70 + "\n")
    input("Press Enter to continue to main menu...")


if __name__ == "__main__":
    try:
        print_quick_info()
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check that all required files are present.")
        input("\nPress Enter to exit...")
