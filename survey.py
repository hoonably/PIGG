import os
import json
from collections import Counter

def print_period_summary(sorted_results):
    """
    Divide the sorted result list into 3 time periods and output
    unique Apps and Scenarios used in each period, along with
    newly added items compared to the previous period.
    """
    print("\n\n--- App and Scenario Summary by Time Period ---")
    n = len(sorted_results)
    if n == 0:
        print("No data to summarize.")
        print("\n------------------------------------")
        return

    # Calculate indices to divide results into 3 periods
    split_points = [0, n // 3, (n * 2) // 3, n]
    
    # Track all Apps and Scenarios from previous periods
    seen_apps = set()
    seen_scenarios = set()

    for i in range(3):
        start_index = split_points[i]
        end_index = split_points[i+1]

        # Skip if current period has no data
        if start_index == end_index:
            continue

        period_data = sorted_results[start_index:end_index]
        
        # Unique Apps and Scenarios in current period (set)
        current_apps = set(item['app'] for item in period_data if item.get('app'))
        current_scenarios = set(item['scenario'] for item in period_data if item.get('scenario'))

        # Find newly added items
        new_apps = sorted(list(current_apps - seen_apps))
        new_scenarios = sorted(list(current_scenarios - seen_scenarios))

        # Output period information (start and end)
        start_ts = period_data[0]['timestamp']
        end_ts = period_data[-1]['timestamp']
        print(f"\n[Period {i+1}: {start_ts} ~ {end_ts}]")
        
        # Output complete unique list
        unique_apps = sorted(list(current_apps))
        unique_scenarios = sorted(list(current_scenarios))
        print(f"  - Unique Apps: {', '.join(unique_apps) if unique_apps else 'N/A'}")
        print(f"  - Unique Scenarios: {', '.join(unique_scenarios) if unique_scenarios else 'N/A'}")

        # Output newly added items (skip first period)
        if i > 0:
            print(f"  - ✨ New Apps: {', '.join(new_apps) if new_apps else 'None'}")
            print(f"  - ✨ New Scenarios: {', '.join(new_scenarios) if new_scenarios else 'None'}")

        # Update seen set with current period items for next period comparison
        seen_apps.update(current_apps)
        seen_scenarios.update(current_scenarios)

    print("\n------------------------------------")


def process_survey_data(root_dir='.'):
    """
    Find folders with timestamp format in the specified root directory,
    organize and summarize the contents of 'survey_result.json' files in chronological order.

    Args:
        root_dir (str): Root directory path containing timestamp folders.
                        Default is current directory.
    """
    results = []

    # Iterate through all items in root directory
    try:
        dir_entries = os.listdir(root_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found: '{root_dir}'")
        return

    for entry_name in dir_entries:
        # Construct full path
        full_path = os.path.join(root_dir, entry_name)

        # Check if item is a directory with timestamp format name
        # (e.g., 20250309_220811)
        if os.path.isdir(full_path) and len(entry_name) == 15 and entry_name[8] == '_':
            # Construct path to survey_result.json file
            json_file_path = os.path.join(full_path, "survey_result.json")

            # Check if JSON file exists
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Extract required data (handle missing keys with None)
                        scenario = data.get("scenario")
                        app = data.get("app")
                        intent_description = data.get("intentDescription")

                        # Add to results list as dictionary
                        results.append({
                            "timestamp": entry_name,
                            "scenario": scenario,
                            "app": app,
                            "intentDescription": intent_description
                        })
                except json.JSONDecodeError:
                    print(f"Warning: '{json_file_path}' file is not in valid JSON format.")
                except Exception as e:
                    print(f"Warning: Error occurred while processing '{json_file_path}': {e}")

    # Sort results chronologically by 'timestamp' key
    # Timestamp string format (YYYYMMDD_HHMMSS) guarantees chronological order when sorted as strings
    sorted_results = sorted(results, key=lambda x: x["timestamp"])

    # Output sorted results
    if not sorted_results:
        print("No data found to analyze. Please check folder structure.")
        return

    # Call function to output summary information by time period
    print_period_summary(sorted_results)


def extract_unique_items(root_dir):
    """
    Analyze the data folder for a specific user and return unique App and Scenario lists (set).
    Also returns usage count for each App/Scenario.
    """
    unique_apps = set()
    unique_scenarios = set()
    app_counts = Counter()
    scenario_counts = Counter()
    total_count = 0

    try:
        dir_entries = os.listdir(root_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{root_dir}' not found.")
        return unique_apps, unique_scenarios, app_counts, scenario_counts, total_count

    for entry_name in dir_entries:
        full_path = os.path.join(root_dir, entry_name)
        if os.path.isdir(full_path) and len(entry_name) == 15 and entry_name[8] == '_':
            json_file_path = os.path.join(full_path, "survey_result.json")
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        total_count += 1
                        if data.get("app"):
                            unique_apps.add(data["app"])
                            app_counts[data["app"]] += 1
                        if data.get("scenario"):
                            unique_scenarios.add(data["scenario"])
                            scenario_counts[data["scenario"]] += 1
                except Exception as e:
                    print(f"Warning: Error occurred while processing '{json_file_path}': {e}")
                    
    return unique_apps, unique_scenarios, app_counts, scenario_counts, total_count


def analyze_all_users(base_dir='.'):
    """
    Scan the specified base directory for numbered user folders,
    analyze and compare App/Scenario usage patterns of all discovered users.
    Save results to survey_result.txt file.
    """
    # List to store output content
    output_lines = []
    
    output_lines.append(f"--- Overall User Usage Pattern Analysis ({base_dir}) ---\n")

    user_data = {}
    try:
        dir_entries = os.listdir(base_dir)
    except FileNotFoundError:
        output_lines.append(f"Error: Directory not found: '{base_dir}'\n")
        return

    # 1. Extract data for each user (sorted numerically)
    for entry_name in sorted(dir_entries, key=lambda x: int(x) if x.isdigit() else 0):
        full_path = os.path.join(base_dir, entry_name)
        if os.path.isdir(full_path) and entry_name.isdigit():
            user_id = entry_name
            apps, scenarios, app_counts, scenario_counts, total = extract_unique_items(full_path)
            if apps or scenarios:  # Add only users with data
                user_data[user_id] = {
                    'apps': apps,
                    'scenarios': scenarios,
                    'app_counts': app_counts,
                    'scenario_counts': scenario_counts,
                    'total_count': total
                }

    if not user_data:
        output_lines.append("No user data found to analyze. Please check folder structure.\n")
        output_lines.append("Base directory should contain numbered folders like '1', '3'.\n")
        # Save to file
        output_file = os.path.join(base_dir, "survey_result.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        print(f"Analysis results saved to '{output_file}'.")
        return

    # Sort user IDs numerically
    sorted_user_ids = sorted(user_data.keys(), key=lambda x: int(x))
    output_lines.append(f"\nAnalyzed users: {', '.join(sorted_user_ids)}\n")

    # 2. Calculate overall statistics and common items
    all_apps = [app for user_id in user_data for app in user_data[user_id]['apps']]
    all_scenarios = [scenario for user_id in user_data for scenario in user_data[user_id]['scenarios']]

    app_counts = Counter(all_apps)
    scenario_counts = Counter(all_scenarios)

    output_lines.append("\n[Most Used Apps] (by number of users)\n")
    if not app_counts:
        output_lines.append("  - No data\n")
    else:
        # Sort alphabetically
        for app in sorted(app_counts.keys()):
            count = app_counts[app]
            output_lines.append(f"  - {app}: {count} users\n")

    output_lines.append("\n[Most Used Scenarios] (by number of users)\n")
    if not scenario_counts:
        output_lines.append("  - No data\n")
    else:
        # Sort alphabetically
        for scenario in sorted(scenario_counts.keys()):
            count = scenario_counts[scenario]
            output_lines.append(f"  - {scenario}: {count} users\n")

    # 3. Analyze unique items per user (by user number)
    output_lines.append("\n[Unique Usage Items per User]\n")
    for user_id in sorted_user_ids:
        data = user_data[user_id]
        # Create set of all other users' Apps/Scenarios
        other_users_apps = set()
        other_users_scenarios = set()
        for other_id, other_data in user_data.items():
            if user_id != other_id:
                other_users_apps.update(other_data['apps'])
                other_users_scenarios.update(other_data['scenarios'])

        # Find unique items for this user through set difference
        unique_to_user_apps = sorted(list(data['apps'] - other_users_apps))
        unique_to_user_scenarios = sorted(list(data['scenarios'] - other_users_scenarios))

        # Calculate total usage count of unique Apps
        unique_app_total_count = sum(data['app_counts'][app] for app in unique_to_user_apps)
        unique_scenario_total_count = sum(data['scenario_counts'][scenario] for scenario in unique_to_user_scenarios)
        
        # Calculate percentage relative to total data
        total_count = data['total_count']
        app_percentage = (unique_app_total_count / total_count * 100) if total_count > 0 else 0
        scenario_percentage = (unique_scenario_total_count / total_count * 100) if total_count > 0 else 0

        output_lines.append(f"\n  -- User '{user_id}' --\n")
        output_lines.append(f"    - Apps unique to this user: {', '.join(unique_to_user_apps) if unique_to_user_apps else 'None'}\n")
        if unique_to_user_apps:
            output_lines.append(f"    - Data: {unique_app_total_count} items ({app_percentage:.1f}%)\n")
        output_lines.append(f"    - Scenarios unique to this user: {', '.join(unique_to_user_scenarios) if unique_to_user_scenarios else 'None'}\n")
        if unique_to_user_scenarios:
            output_lines.append(f"    - Data: {unique_scenario_total_count} items ({scenario_percentage:.1f}%)\n")

    output_lines.append("\n----------------------------------------------------\n")
    
    # Save to file
    output_file = os.path.join("./survey_result.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"Analysis results saved to '{output_file}'.")
    
if __name__ == "__main__":
    # Process data based on the script execution location.
    # Modify the paths below if you want to specify a different folder.
    process_survey_data('./dataset/fingertip-20k/1')
    process_survey_data('./dataset/fingertip-20k/2')
    process_survey_data('./dataset/fingertip-20k/3')
    process_survey_data('./dataset/fingertip-20k/4')
    process_survey_data('./dataset/fingertip-20k/5')
    process_survey_data('./dataset/fingertip-20k/6')
    process_survey_data('./dataset/fingertip-20k/7')
    process_survey_data('./dataset/fingertip-20k/8')
    process_survey_data('./dataset/fingertip-20k/9')
    process_survey_data('./dataset/fingertip-20k/10')

    
    analyze_all_users('./dataset/fingertip-20k/')