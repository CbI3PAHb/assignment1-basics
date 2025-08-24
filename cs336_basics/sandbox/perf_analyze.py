# import pstats

# perf_file_path = "/Users/parii-artem/Documents/assignment1-basics/output.prof"

# stats = pstats.Stats(perf_file_path) # Load the profiling data
# stats.sort_stats('tottime') # Sort by total time spent in the function itself
# stats.print_stats() # Print all statistics
# # stats.print_callers() # Show who called which functions
# # stats.print_callees() # Show which functions were called by others


# analyze_profile.py
import pstats

# Load the profile data
perf_file_path = "/Users/parii-artem/Documents/assignment1-basics/output.prof"

p = pstats.Stats(perf_file_path)
n_top_stats = 32

# Sort by different criteria and print results
print(f"=== Top {n_top_stats} by cumulative time ===")
p.sort_stats('cumulative').print_stats(n_top_stats)

print(f"\n=== Top {n_top_stats} by internal time ===")
p.sort_stats('time').print_stats(n_top_stats)

print(f"\n=== Top {n_top_stats} by number of calls ===")
p.sort_stats('calls').print_stats(n_top_stats)
