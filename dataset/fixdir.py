import os
import shutil

root = "./fingertip-20k"

for uid in os.listdir(root):
    uid_path = os.path.join(root, uid)
    if not os.path.isdir(uid_path):
        continue
    
    nested = os.path.join(uid_path, uid)  # dataset/uid/uid
    if os.path.isdir(nested):
        # 안에 있는 파일/폴더를 바깥으로 이동
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(uid_path, item)
            if os.path.exists(dst):
                print(f"⚠️ Skip (already exists): {dst}")
                continue
            shutil.move(src, dst)
        
        # 빈 nested 폴더 삭제
        os.rmdir(nested)
        print(f"✔ Fixed: {nested} -> {uid_path}")
