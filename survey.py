import os
import json
from collections import Counter

def print_period_summary(sorted_results):
    """
    정렬된 결과 리스트를 3개의 시간대로 나누어 각 시간대별로
    사용된 App과 Scenario의 중복 없는 목록과, 이전 시간대 대비
    새롭게 추가된 항목을 출력합니다.
    """
    print("\n\n--- 시간대별 App 및 Scenario 요약 ---")
    n = len(sorted_results)
    if n == 0:
        print("요약할 데이터가 없습니다.")
        print("\n------------------------------------")
        return

    # 결과를 3개의 구간으로 나눌 인덱스를 계산합니다.
    split_points = [0, n // 3, (n * 2) // 3, n]
    
    # 이전 시간대까지의 모든 App과 Scenario를 추적하기 위한 set
    seen_apps = set()
    seen_scenarios = set()

    for i in range(3):
        start_index = split_points[i]
        end_index = split_points[i+1]

        # 현재 시간대의 데이터가 없으면 건너뜁니다.
        if start_index == end_index:
            continue

        period_data = sorted_results[start_index:end_index]
        
        # 현재 시간대의 중복 없는 App과 Scenario (set)
        current_apps = set(item['app'] for item in period_data if item.get('app'))
        current_scenarios = set(item['scenario'] for item in period_data if item.get('scenario'))

        # 새롭게 추가된 항목 찾기
        new_apps = sorted(list(current_apps - seen_apps))
        new_scenarios = sorted(list(current_scenarios - seen_scenarios))

        # 시간대 정보(시작과 끝)를 출력합니다.
        start_ts = period_data[0]['timestamp']
        end_ts = period_data[-1]['timestamp']
        print(f"\n[시간대 {i+1}: {start_ts} ~ {end_ts}]")
        
        # 전체 고유 목록 출력
        unique_apps = sorted(list(current_apps))
        unique_scenarios = sorted(list(current_scenarios))
        print(f"  - Unique Apps: {', '.join(unique_apps) if unique_apps else 'N/A'}")
        print(f"  - Unique Scenarios: {', '.join(unique_scenarios) if unique_scenarios else 'N/A'}")

        # 새롭게 추가된 항목 출력 (첫 시간대는 제외)
        if i > 0:
            print(f"  - ✨ New Apps: {', '.join(new_apps) if new_apps else '없음'}")
            print(f"  - ✨ New Scenarios: {', '.join(new_scenarios) if new_scenarios else '없음'}")

        # 현재 시간대의 항목들을 seen set에 추가하여 다음 시간대와 비교할 수 있도록 업데이트
        seen_apps.update(current_apps)
        seen_scenarios.update(current_scenarios)

    print("\n------------------------------------")


def process_survey_data(root_dir='.'):
    """
    지정된 루트 디렉토리에서 타임스탬프 형식의 폴더를 찾아
    'survey_result.json' 파일의 내용을 시간순으로 정리하고 요약합니다.

    Args:
        root_dir (str): 타임스탬프 폴더들이 있는 루트 디렉토리 경로.
                        기본값은 현재 디렉토리입니다.
    """
    results = []

    # 루트 디렉토리의 모든 항목을 순회합니다.
    try:
        dir_entries = os.listdir(root_dir)
    except FileNotFoundError:
        print(f"오류: 디렉토리를 찾을 수 없습니다: '{root_dir}'")
        return

    for entry_name in dir_entries:
        # 항목의 전체 경로를 구성합니다.
        full_path = os.path.join(root_dir, entry_name)

        # 해당 항목이 디렉토리인지, 이름이 타임스탬프 형식인지 확인합니다.
        # (예: 20250309_220811)
        if os.path.isdir(full_path) and len(entry_name) == 15 and entry_name[8] == '_':
            # survey_result.json 파일의 경로를 구성합니다.
            json_file_path = os.path.join(full_path, "survey_result.json")

            # JSON 파일이 존재하는지 확인합니다.
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 필요한 데이터 추출 (키가 없는 경우 None으로 처리)
                        scenario = data.get("scenario")
                        app = data.get("app")
                        intent_description = data.get("intentDescription")

                        # 결과 리스트에 딕셔너리 형태로 추가
                        results.append({
                            "timestamp": entry_name,
                            "scenario": scenario,
                            "app": app,
                            "intentDescription": intent_description
                        })
                except json.JSONDecodeError:
                    print(f"경고: '{json_file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
                except Exception as e:
                    print(f"경고: '{json_file_path}' 파일을 처리하는 중 오류 발생: {e}")

    # 'timestamp' 키를 기준으로 결과를 시간순으로 정렬합니다.
    # 타임스탬프 문자열 형식(YYYYMMDD_HHMMSS)은 문자열 정렬 시 시간순서가 보장됩니다.
    sorted_results = sorted(results, key=lambda x: x["timestamp"])

    # 정렬된 결과를 출력합니다.
    if not sorted_results:
        print("분석할 데이터를 찾지 못했습니다. 폴더 구조를 확인해주세요.")
        return

    # 시간대별 요약 정보를 출력하는 함수를 호출합니다.
    print_period_summary(sorted_results)


def extract_unique_items(root_dir):
    """
    지정된 사용자의 데이터 폴더를 분석하여 고유한 App과 Scenario 목록(set)을 반환합니다.
    """
    unique_apps = set()
    unique_scenarios = set()

    try:
        dir_entries = os.listdir(root_dir)
    except FileNotFoundError:
        print(f"오류: '{root_dir}' 디렉토리를 찾을 수 없습니다.")
        return unique_apps, unique_scenarios

    for entry_name in dir_entries:
        full_path = os.path.join(root_dir, entry_name)
        if os.path.isdir(full_path) and len(entry_name) == 15 and entry_name[8] == '_':
            json_file_path = os.path.join(full_path, "survey_result.json")
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("app"):
                            unique_apps.add(data["app"])
                        if data.get("scenario"):
                            unique_scenarios.add(data["scenario"])
                except Exception as e:
                    print(f"경고: '{json_file_path}' 처리 중 오류 발생: {e}")
                    
    return unique_apps, unique_scenarios


def analyze_all_users(base_dir='.'):
    """
    지정된 기본 디렉토리에서 숫자로 된 사용자 폴더를 스캔하여,
    발견된 모든 사용자들의 App/Scenario 사용 패턴을 분석하고 비교합니다.
    """
    print(f"\n\n--- 전체 사용자 이용 패턴 분석 ({base_dir}) ---")

    user_data = {}
    try:
        dir_entries = os.listdir(base_dir)
    except FileNotFoundError:
        print(f"오류: 디렉토리를 찾을 수 없습니다: '{base_dir}'")
        return

    # 1. 각 사용자별 데이터 추출
    for entry_name in dir_entries:
        full_path = os.path.join(base_dir, entry_name)
        if os.path.isdir(full_path) and entry_name.isdigit():
            user_id = entry_name
            apps, scenarios = extract_unique_items(full_path)
            if apps or scenarios:  # 데이터가 있는 사용자만 추가
                user_data[user_id] = {
                    'apps': apps,
                    'scenarios': scenarios
                }

    if not user_data:
        print("분석할 사용자 데이터를 찾지 못했습니다. 폴더 구조를 확인해주세요.")
        print("베이스 디렉토리 안에 '1', '3'과 같이 숫자로 된 폴더가 있어야 합니다.")
        return

    print(f"\n분석 대상 사용자: {', '.join(sorted(user_data.keys()))}")

    # 2. 전체 통계 및 공통 항목 계산
    all_apps = [app for user_id in user_data for app in user_data[user_id]['apps']]
    all_scenarios = [scenario for user_id in user_data for scenario in user_data[user_id]['scenarios']]

    app_counts = Counter(all_apps)
    scenario_counts = Counter(all_scenarios)

    print("\n[가장 많이 사용된 App] (사용자 수 기준)")
    if not app_counts:
        print("  - 데이터 없음")
    else:
        for app, count in app_counts.most_common():
            print(f"  - {app}: {count}명")

    print("\n[가장 많이 사용된 Scenario] (사용자 수 기준)")
    if not scenario_counts:
        print("  - 데이터 없음")
    else:
        for scenario, count in scenario_counts.most_common():
            print(f"  - {scenario}: {count}명")

    # 3. 사용자별 고유 항목 분석
    print("\n[사용자별 고유 사용 항목]")
    for user_id, data in user_data.items():
        # 자신을 제외한 다른 모든 사용자의 App/Scenario 집합 생성
        other_users_apps = set()
        other_users_scenarios = set()
        for other_id, other_data in user_data.items():
            if user_id != other_id:
                other_users_apps.update(other_data['apps'])
                other_users_scenarios.update(other_data['scenarios'])

        # 차집합을 통해 해당 사용자만 사용하는 고유 항목 찾기
        unique_to_user_apps = sorted(list(data['apps'] - other_users_apps))
        unique_to_user_scenarios = sorted(list(data['scenarios'] - other_users_scenarios))

        print(f"\n  -- 사용자 '{user_id}' --")
        print(f"    - 이 사용자만 사용하는 App: {', '.join(unique_to_user_apps) if unique_to_user_apps else '없음'}")
        print(f"    - 이 사용자만 사용하는 Scenario: {', '.join(unique_to_user_scenarios) if unique_to_user_scenarios else '없음'}")

    print("\n----------------------------------------------------")
    
if __name__ == "__main__":
    # 스크립트가 실행되는 위치를 기준으로 데이터를 처리합니다.
    # 다른 폴더를 지정하고 싶다면 아래 경로를 수정하세요.
    # 예: process_survey_data('/path/to/your/data_folder')
    process_survey_data('./data')
    # analyze_all_users('./data')