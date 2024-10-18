#!/bin/bash

# 遍历所有符合条件的子目录
for dir in */; do
    if [[ -d "$dir" ]]; then  # 检查是否为目录
        # 提取 ID
        id=$(basename "$dir" | grep -oP '^\d{6}')  # 获取目录名中的六位数字作为 ID
	echo $id
        
        # 创建新的 ID 子目录
        mkdir -p "$id"  # 创建 ID 目录，如果已存在则不报错
        
        # 遍历子目录中的 JPG 文件
        for file in "$dir"/*.jpg; do
            if [[ -f "$file" ]]; then  # 检查是否为文件
                # 提取帧 ID
                frame_id=$(echo "$file" | grep -oP '\d{4}' | tail -n 1)  # 获取最后四位数字
                # 新文件名
                new_name="${id}_${frame_id}.jpg"
                # 移动并重命名文件到新的 ID 目录
                mv "$file" "$id/$new_name"
            fi
        done
    fi
done

